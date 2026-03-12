"""
search/agent.py — Agentic search and RAG controller for Global News Pulse.

This module is the "brain" of the application.  It coordinates the embedding
engine, the Endee vector database, and the LLM to deliver two modes of
operation:

Simple Search
-------------
A direct semantic lookup: embed the query → ANN search Endee → return ranked
results with source metadata.  Fast (< 1 s), no LLM call.

Agentic Search
--------------
A multi-step intelligence workflow:

  1. **Initial retrieval** — semantic search for the topic (top 10).
  2. **Diversity check** — if ≥ 60 % of results share a single source, the
     agent automatically widens the search to surface differing perspectives.
  3. **Sub-trend extraction** — the LLM analyses the retrieved headlines and
     identifies 2–4 specific sub-trends or emerging angles.
  4. **Secondary retrieval** — a targeted Endee search is run for each
     sub-trend to gather deeper, more specific evidence.
  5. **Synthesis** — all evidence (with URL + source citations) is fed to the
     LLM, which produces a structured intelligence brief in markdown.

Source Grounding
----------------
Every piece of evidence carries its ``url``, ``source``, and ``title`` from
the Endee metadata store.  The synthesis prompt explicitly instructs the LLM
to cite these inline.  The :class:`AgenticBrief` also returns a deduplicated
``sources`` manifest so the UI can render a clean citation index.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import llm_cfg, LLMConfig
from database_manager import DatabaseManager, EndeeConnectionError, EndeeSearchError
from embeddings.manager import EmbeddingManager
from .llm_client import LLMClient, ANALYST_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class AgenticBrief:
    """
    The structured output of :meth:`NewsAgent.agentic_search`.

    Attributes
    ----------
    topic:
        The original user query / topic.
    summary:
        LLM-generated intelligence brief in markdown format.
    sub_trends:
        List of specific sub-trends identified from the initial results.
    sources:
        Deduplicated citation index — each entry is a dict with
        ``title``, ``url``, ``source``, ``similarity``, ``published_at``.
    initial_results:
        Raw Endee results from the first-pass semantic search.
    secondary_evidence:
        Mapping of sub-trend string → list of Endee results for that trend.
    diversity_expanded:
        ``True`` if the agent triggered the diversity expansion step.
    elapsed_seconds:
        Total wall-clock time for the full agentic pipeline.
    """

    topic: str
    summary: str
    sub_trends: list[str]
    sources: list[dict[str, Any]]
    initial_results: list[dict[str, Any]]
    secondary_evidence: dict[str, list[dict[str, Any]]]
    diversity_expanded: bool
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SUB_TREND_PROMPT = """\
The following news article titles were retrieved for the topic: "{topic}"

Article titles:
{titles}

Your task: Identify exactly 2 to 4 specific, distinct sub-trends, emerging \
angles, or underexplored dimensions within this topic that deserve deeper \
investigation.

Return ONLY a valid JSON object — no prose, no markdown fences:
{{"sub_trends": ["sub-trend 1", "sub-trend 2", "sub-trend 3"]}}
"""

_SYNTHESIS_PROMPT = """\
You are briefing the intelligence team on the topic: **{topic}**

EVIDENCE BASE ({article_count} articles, {source_count} unique sources):
{evidence_block}

IDENTIFIED SUB-TRENDS: {sub_trends_str}

DIVERSITY NOTE: {diversity_note}

---

Produce a comprehensive intelligence brief using this exact structure:

## Executive Summary
(2–3 sentence high-altitude overview; most impactful finding first)

## Key Developments
(Bullet list of confirmed facts, each grounded with [Source](URL) citation)

## Sub-Trend Analysis
(One paragraph per sub-trend; include supporting evidence and implications)

## Divergent Perspectives
(Highlight contradictions, competing narratives, or under-reported angles across sources)

## Signal Assessment
(Rate each sub-trend: Confirmed / Inferred / Speculative — with brief justification)

## Source Index
(Numbered list: [N] Title — Source — URL)
"""

# ---------------------------------------------------------------------------
# NewsAgent
# ---------------------------------------------------------------------------


class NewsAgent:
    """
    Orchestrates semantic search, agentic sub-trend analysis, and LLM synthesis.

    Parameters
    ----------
    db:
        Optional :class:`~database_manager.DatabaseManager` instance.  A new
        one is created if not supplied.
    llm:
        Optional :class:`~search.llm_client.LLMClient` instance.
    config:
        An :class:`~config.LLMConfig` instance for diversity thresholds.

    Notes
    -----
    The :class:`~embeddings.manager.EmbeddingManager` is initialised lazily on
    the first search call because the sentence-transformer model is ~100 MB and
    slow to load.  Subsequent calls reuse the cached instance.

    Examples
    --------
    >>> agent = NewsAgent()
    >>> results = agent.simple_search("AI regulation EU")
    >>> brief = agent.agentic_search("AI regulation EU")
    >>> print(brief.summary)
    """

    def __init__(
        self,
        db: DatabaseManager | None = None,
        llm: LLMClient | None = None,
        config: LLMConfig = llm_cfg,
    ) -> None:
        self._cfg = config
        self._db = db or DatabaseManager()
        self._llm = llm or LLMClient()
        self._embedder: EmbeddingManager | None = None  # lazy init

    # ------------------------------------------------------------------
    # Lazy embedding manager
    # ------------------------------------------------------------------

    @property
    def _embed(self) -> EmbeddingManager:
        """Return the :class:`EmbeddingManager`, initialising it on first access."""
        if self._embedder is None:
            self._embedder = EmbeddingManager()
        return self._embedder

    # ------------------------------------------------------------------
    # Internal: search helpers
    # ------------------------------------------------------------------

    def _search_by_text(
        self,
        query: str,
        top_k: int = 10,
        *,
        source_filter: str | None = None,
        category_filter: str | None = None,
        ef: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Encode *query* and run a similarity search against Endee.

        Parameters
        ----------
        query:
            Natural language search query.
        top_k:
            Number of nearest neighbours to return.
        source_filter:
            Restrict results to a specific news outlet.
        category_filter:
            Restrict results to a specific topic category.
        ef:
            HNSW runtime search depth (0 = server default).

        Returns
        -------
        list[dict]
            Ranked Endee results with all metadata fields.
        """
        vector = self._embed.encode_single(query)
        try:
            results = self._db.similarity_search(
                vector,
                top_k=top_k,
                source_filter=source_filter,
                category_filter=category_filter,
                ef=ef,
            )
        except (EndeeConnectionError, EndeeSearchError) as exc:
            logger.error("Endee search failed for query '%s': %s", query, exc)
            results = []

        if not results:
            logger.info(
                "Endee returned 0 results for '%s' — falling back to live NewsAPI search.",
                query,
            )
            results = self._live_search(query, top_k=top_k)

        return results

    # ------------------------------------------------------------------
    # Fallback: live NewsAPI + in-memory cosine similarity
    # ------------------------------------------------------------------

    def _live_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Fetch live articles from NewsAPI for *query* and rank them by cosine
        similarity to the encoded query vector.

        Called automatically when Endee is unreachable or its index is empty.
        Requires NEWS_API_KEY to be configured.
        """
        from ingestion.news_provider import NewsProvider

        try:
            provider = NewsProvider()
        except ValueError as exc:
            logger.warning("Live search unavailable (no NEWS_API_KEY): %s", exc)
            return []

        try:
            articles = provider.fetch_articles(query, sort_by="relevancy")
        except RuntimeError as exc:
            logger.error("NewsAPI fetch failed for '%s': %s", query, exc)
            return []

        if not articles:
            return []

        # Embed query and all article texts
        texts = [a["embed_text"] for a in articles]
        query_vec = np.array(self._embed.encode_single(query), dtype=np.float32)
        article_vecs = np.array(self._embed.encode(texts), dtype=np.float32)

        # Cosine similarity == dot product because fastembed normalises to unit sphere
        scores: np.ndarray = article_vecs @ query_vec

        top_idxs = np.argsort(scores)[::-1][:top_k]

        results: list[dict[str, Any]] = []
        for i in top_idxs:
            a = articles[int(i)]
            results.append({
                "id":           a["id"],
                "similarity":   float(scores[i]),
                "title":        a.get("title", ""),
                "url":          a.get("url", ""),
                "source":       a.get("source", ""),
                "category":     a.get("category", ""),
                "published_at": a.get("published_at", ""),
                "description":  a.get("description", ""),
                "content":      a.get("content", ""),
                "author":       a.get("author", ""),
            })

        logger.info(
            "Live search returned %d result(s) for '%s' (NEWS_API fallback).",
            len(results), query,
        )
        return results

    # ------------------------------------------------------------------
    # Internal: diversity check
    # ------------------------------------------------------------------

    def _ensure_source_diversity(
        self,
        topic: str,
        results: list[dict[str, Any]],
        initial_top_k: int = 10,
    ) -> tuple[list[dict[str, Any]], bool]:
        """
        Check whether results are concentrated in a single source and, if so,
        expand the search pool to surface differing perspectives.

        The diversity ratio is ``len(unique_sources) / len(results)``.  When
        this falls below ``config.diversity_threshold`` (default 0.4), the
        agent fetches 30 results and merges novel articles into the pool,
        keeping the top ``initial_top_k`` by similarity.

        Parameters
        ----------
        topic:
            The search query used for the expanded fetch.
        results:
            The current result list to evaluate.
        initial_top_k:
            Maximum size of the returned list.

        Returns
        -------
        tuple[list[dict], bool]
            Updated result list and a flag indicating whether expansion
            was triggered.
        """
        if not results:
            return results, False

        unique_sources = {r["source"] for r in results if r.get("source")}
        if not unique_sources:
            return results, False

        diversity_ratio = len(unique_sources) / len(results)
        logger.info(
            "Diversity check: %d unique source(s) / %d result(s) = %.2f "
            "(threshold %.2f)",
            len(unique_sources),
            len(results),
            diversity_ratio,
            self._cfg.diversity_threshold,
        )

        if diversity_ratio >= self._cfg.diversity_threshold:
            return results, False

        # Trigger expansion
        dominant_source = max(
            unique_sources,
            key=lambda s: sum(1 for r in results if r.get("source") == s),
        )
        logger.warning(
            "Low source diversity detected (dominant source: '%s').  "
            "Expanding search to top 30 for broader perspectives.",
            dominant_source,
        )

        expanded = self._search_by_text(topic, top_k=30, ef=200)

        # Merge: keep originals + novel articles from expansion, dedup by ID
        seen_ids = {r["id"] for r in results}
        for article in expanded:
            if article["id"] not in seen_ids:
                results.append(article)
                seen_ids.add(article["id"])

        # Re-sort by similarity (descending) and trim to initial_top_k
        results.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)
        results = results[:initial_top_k]

        new_sources = {r["source"] for r in results if r.get("source")}
        logger.info(
            "After expansion: %d unique source(s) across %d results.",
            len(new_sources),
            len(results),
        )
        return results, True

    # ------------------------------------------------------------------
    # Internal: sub-trend extraction via LLM
    # ------------------------------------------------------------------

    def _extract_sub_trends(
        self,
        topic: str,
        results: list[dict[str, Any]],
    ) -> list[str]:
        """
        Ask the LLM to identify 2–4 specific sub-trends from retrieved titles.

        Parameters
        ----------
        topic:
            The main search topic.
        results:
            Initial Endee results; only ``title`` is used.

        Returns
        -------
        list[str]
            Between 2 and 4 sub-trend strings.  Falls back to a single generic
            sub-trend if the LLM call fails or returns malformed JSON.
        """
        titles = "\n".join(
            f"- {r['title']}"
            for r in results
            if r.get("title") and r["title"] != "[Removed]"
        )
        if not titles:
            return [topic]

        prompt = _SUB_TREND_PROMPT.format(topic=topic, titles=titles)
        try:
            raw = self._llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
                system_prompt="You are a precise JSON-generating assistant.  "
                              "Output ONLY valid JSON, nothing else.",
            )
            parsed = self._llm.extract_json(raw)
            sub_trends = parsed.get("sub_trends", [])
            if isinstance(sub_trends, list) and sub_trends:
                # Sanitise: strings only, max 4, strip whitespace
                sub_trends = [str(t).strip() for t in sub_trends[:4] if t]
                logger.info("Sub-trends identified: %s", sub_trends)
                return sub_trends
        except Exception as exc:
            logger.warning(
                "Sub-trend extraction failed (%s).  Falling back to single trend.", exc
            )

        return [topic]

    # ------------------------------------------------------------------
    # Internal: evidence block builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence_block(
        initial_results: list[dict[str, Any]],
        secondary_evidence: dict[str, list[dict[str, Any]]],
    ) -> str:
        """
        Render all retrieved articles into a structured text block for the
        synthesis LLM prompt.

        Format per article::

            [Article N]
            Title      : ...
            Source     : ...
            URL        : ...
            Similarity : 0.924
            Description: ...
            Sub-trend  : ... (omitted for initial results)

        Parameters
        ----------
        initial_results:
            First-pass Endee results.
        secondary_evidence:
            Mapping of sub-trend → Endee results.

        Returns
        -------
        str
            Multi-line evidence block.
        """
        lines: list[str] = []
        n = 1

        for article in initial_results:
            lines.extend(_format_article(article, n))
            n += 1

        for sub_trend, articles in secondary_evidence.items():
            for article in articles:
                lines.extend(_format_article(article, n, sub_trend=sub_trend))
                n += 1

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: synthesis via LLM
    # ------------------------------------------------------------------

    def _synthesize_brief(
        self,
        topic: str,
        initial_results: list[dict[str, Any]],
        sub_trends: list[str],
        secondary_evidence: dict[str, list[dict[str, Any]]],
        diversity_expanded: bool,
    ) -> str:
        """
        Assemble all evidence and ask the LLM to produce the intelligence brief.

        Parameters
        ----------
        topic:
            Main search topic.
        initial_results:
            First-pass results.
        sub_trends:
            Identified sub-trends.
        secondary_evidence:
            Sub-trend → secondary results mapping.
        diversity_expanded:
            Whether the diversity expansion step was triggered.

        Returns
        -------
        str
            Markdown-formatted intelligence brief.
        """
        # Deduplicate all articles by ID for the evidence block
        all_articles: list[dict] = list(initial_results)
        seen_ids = {r["id"] for r in initial_results}
        for articles in secondary_evidence.values():
            for a in articles:
                if a["id"] not in seen_ids:
                    all_articles.append(a)
                    seen_ids.add(a["id"])

        all_sources = {r["source"] for r in all_articles if r.get("source")}
        evidence_block = self._build_evidence_block(initial_results, secondary_evidence)

        diversity_note = (
            "The agent detected low source diversity in the initial results and "
            "automatically expanded the search to include additional perspectives."
            if diversity_expanded
            else "Source distribution is adequate; no expansion was required."
        )

        prompt = _SYNTHESIS_PROMPT.format(
            topic=topic,
            article_count=len(all_articles),
            source_count=len(all_sources),
            evidence_block=evidence_block,
            sub_trends_str=", ".join(sub_trends),
            diversity_note=diversity_note,
        )

        try:
            brief = self._llm.complete(
                [{"role": "user", "content": prompt}],
                system_prompt=ANALYST_SYSTEM_PROMPT,
            )
            return brief
        except RuntimeError as exc:
            logger.error("Brief synthesis failed: %s", exc)
            return (
                f"## Intelligence Brief — {topic}\n\n"
                "**Note:** LLM synthesis unavailable.  "
                "Raw search results are available below.\n\n"
                f"*Error: {exc}*"
            )

    # ------------------------------------------------------------------
    # Internal: citation index builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sources(
        initial_results: list[dict[str, Any]],
        secondary_evidence: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """
        Build a deduplicated citation index from all retrieved articles.

        Returns
        -------
        list[dict]
            Unique articles sorted by similarity (descending).  Each entry
            contains ``id``, ``title``, ``url``, ``source``,
            ``similarity``, ``published_at``.
        """
        seen: set[str] = set()
        sources: list[dict[str, Any]] = []

        for article in initial_results:
            if article["id"] not in seen:
                sources.append(_citation_entry(article))
                seen.add(article["id"])

        for articles in secondary_evidence.values():
            for article in articles:
                if article["id"] not in seen:
                    sources.append(_citation_entry(article))
                    seen.add(article["id"])

        sources.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        return sources

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simple_search(self, query: str) -> list[dict[str, Any]]:
        """
        Perform a fast semantic search and return the top 5 results.

        No LLM calls are made.  Latency is dominated by the embedding encode
        (< 50 ms on CPU) and the Endee ANN query (< 10 ms).

        Parameters
        ----------
        query:
            Natural language search query.

        Returns
        -------
        list[dict]
            Up to 5 results, each containing ``id``, ``similarity``,
            ``title``, ``url``, ``source``, ``category``,
            ``published_at``, ``description``.
        """
        logger.info("Simple search: '%s'", query)
        results = self._search_by_text(query, top_k=5)
        logger.info("Simple search returned %d result(s).", len(results))
        return results

    def agentic_search(self, topic: str) -> AgenticBrief:
        """
        Run the full multi-step agentic intelligence pipeline for *topic*.

        Pipeline
        --------
        1. Initial semantic search (top 10) via Endee.
        2. Source diversity check — auto-expand if needed.
        3. Sub-trend extraction via LLM (structured JSON).
        4. Secondary Endee searches for each sub-trend (top 4 each).
        5. LLM synthesis into a structured intelligence brief with citations.

        Parameters
        ----------
        topic:
            The news topic or question to analyse.

        Returns
        -------
        AgenticBrief
            Full structured output including the markdown brief, citation
            index, sub-trends, and raw evidence.
        """
        t_start = time.monotonic()
        logger.info("Agentic search started: '%s'", topic)

        # ── Step 1: Initial retrieval ─────────────────────────────────────────
        initial_results = self._search_by_text(topic, top_k=10, ef=200)
        logger.info("Step 1 complete: %d initial result(s).", len(initial_results))

        if not initial_results:
            elapsed = round(time.monotonic() - t_start, 2)
            return AgenticBrief(
                topic=topic,
                summary=(
                    f"## No results found for '{topic}'\n\n"
                    "Could not retrieve articles from Endee or NewsAPI.\n"
                    "Check that NEWS_API_KEY is configured."
                ),
                sub_trends=[],
                sources=[],
                initial_results=[],
                secondary_evidence={},
                diversity_expanded=False,
                elapsed_seconds=elapsed,
            )

        # ── Step 2: Diversity check ───────────────────────────────────────────
        initial_results, diversity_expanded = self._ensure_source_diversity(
            topic, initial_results
        )
        logger.info(
            "Step 2 complete: diversity_expanded=%s.", diversity_expanded
        )

        # ── Step 3: Sub-trend extraction ──────────────────────────────────────
        sub_trends = self._extract_sub_trends(topic, initial_results)
        logger.info("Step 3 complete: sub_trends=%s.", sub_trends)

        # ── Step 4: Secondary retrieval per sub-trend ─────────────────────────
        secondary_evidence: dict[str, list[dict[str, Any]]] = {}
        for trend in sub_trends:
            trend_results = self._search_by_text(trend, top_k=4)
            # Exclude articles already in the initial set to avoid redundancy
            initial_ids = {r["id"] for r in initial_results}
            novel = [r for r in trend_results if r["id"] not in initial_ids]
            secondary_evidence[trend] = novel
            logger.info(
                "Step 4 — sub-trend '%s': %d novel result(s).", trend, len(novel)
            )

        # ── Step 5: LLM synthesis ─────────────────────────────────────────────
        summary = self._synthesize_brief(
            topic, initial_results, sub_trends, secondary_evidence, diversity_expanded
        )
        logger.info("Step 5 complete: brief synthesised.")

        # ── Assemble final output ─────────────────────────────────────────────
        sources = self._build_sources(initial_results, secondary_evidence)
        elapsed = round(time.monotonic() - t_start, 2)

        logger.info(
            "Agentic search complete in %.2fs — %d sources cited.", elapsed, len(sources)
        )

        return AgenticBrief(
            topic=topic,
            summary=summary,
            sub_trends=sub_trends,
            sources=sources,
            initial_results=initial_results,
            secondary_evidence=secondary_evidence,
            diversity_expanded=diversity_expanded,
            elapsed_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------


def _format_article(
    article: dict[str, Any],
    n: int,
    *,
    sub_trend: str | None = None,
) -> list[str]:
    """Render a single article as a numbered evidence block for the LLM prompt."""
    lines = [
        f"[Article {n}]",
        f"Title      : {article.get('title', 'N/A')}",
        f"Source     : {article.get('source', 'N/A')}",
        f"URL        : {article.get('url', 'N/A')}",
        f"Similarity : {article.get('similarity', 0.0):.3f}",
        f"Published  : {article.get('published_at', 'N/A')}",
        f"Description: {article.get('description', '') or '(none)'}",
    ]
    if sub_trend:
        lines.append(f"Sub-trend  : {sub_trend}")
    lines.append("")  # blank line separator
    return lines


def _citation_entry(article: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields needed for the citation index."""
    return {
        "id":           article.get("id", ""),
        "title":        article.get("title", ""),
        "url":          article.get("url", ""),
        "source":       article.get("source", ""),
        "similarity":   article.get("similarity", 0.0),
        "published_at": article.get("published_at", ""),
        "description":  article.get("description", ""),
    }
