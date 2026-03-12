"""
ingestion/news_provider.py — NewsAPI data fetcher and article formatter.

This module is responsible for fetching, cleaning, deduplicating, and
structuring raw news articles into the format expected by
:class:`~database_manager.DatabaseManager`.

Metadata strategy recap
-----------------------
``meta`` (str → stored as UTF-8 bytes in Endee)
    A JSON string containing the full article payload:
    ``{title, url, source, description, content, published_at, category}``.
    The C++ server receives this as a string and stores it verbatim as
    ``std::vector<uint8_t>`` bytes (see ``main.cpp``).
    On retrieval the bytes are decoded back to UTF-8 and parsed as JSON.

``filter`` (str)
    A flat JSON string of low-cardinality categorical fields that Endee
    indexes for fast filtered ANN search:
    ``{"source": "TechCrunch", "category": "AI"}``.
    Consumers can query with ``[{"source": {"$eq": "TechCrunch"}}]``.

``embed_text`` (str, ephemeral)
    Title + description concatenation that is fed to the embedding model.
    This field is NOT stored in Endee — it is consumed by the ingestion
    pipeline and stripped before upsert.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import requests

from config import news_cfg, NewsAPIConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# Pattern that strips the "[+NNN chars]" truncation marker NewsAPI appends
_TRUNCATION_PATTERN = re.compile(r"\[\+\d+ chars?\]", re.IGNORECASE)

# Generic HTML tag pattern for light sanitisation
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>", re.DOTALL)


# ---------------------------------------------------------------------------
# NewsProvider
# ---------------------------------------------------------------------------

class NewsProvider:
    """
    Fetches and formats news articles from the NewsAPI ``/v2/everything``
    endpoint.

    Parameters
    ----------
    config:
        A :class:`~config.NewsAPIConfig` instance.  Defaults to the
        module-level singleton read from environment variables.

    Examples
    --------
    >>> provider = NewsProvider()
    >>> articles = provider.fetch_all_topics()
    >>> print(articles[0].keys())
    dict_keys(['id', 'embed_text', 'title', 'url', ...])
    """

    def __init__(self, config: NewsAPIConfig = news_cfg) -> None:
        if not config.api_key:
            raise ValueError(
                "NEWS_API_KEY is not set.  "
                "Copy .env.example to .env and fill in your NewsAPI key."
            )
        self._cfg = config
        self._session = self._build_session()
        logger.info(
            "NewsProvider initialised — topics: %s, page_size: %d",
            self._cfg.topics,
            self._cfg.page_size,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """Build an HTTP session with the NewsAPI key in the header."""
        session = requests.Session()
        session.headers.update({"X-Api-Key": self._cfg.api_key})
        return session

    # ------------------------------------------------------------------
    # Text cleaning
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str | None) -> str:
        """
        Sanitise a raw text string from the NewsAPI response.

        Operations performed (in order):
        1. Return an empty string for ``None`` or blank input.
        2. Strip all HTML tags (``<p>``, ``<br/>``, etc.).
        3. Remove NewsAPI's ``[+NNN chars]`` truncation marker.
        4. Collapse multiple whitespace sequences into a single space.
        5. Strip leading/trailing whitespace.

        Parameters
        ----------
        text:
            Raw string from the NewsAPI JSON payload.

        Returns
        -------
        str
            Cleaned, normalised text.
        """
        if not text:
            return ""
        text = _HTML_TAG_PATTERN.sub(" ", text)
        text = _TRUNCATION_PATTERN.sub("", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _stable_id(url: str) -> str:
        """
        Derive a deterministic vector ID from the article URL.

        Using an MD5 hash of the URL means that re-ingesting the same article
        results in an *upsert* (overwrite) rather than a duplicate insert.

        Parameters
        ----------
        url:
            The canonical article URL.

        Returns
        -------
        str
            A 32-character hex string.
        """
        return hashlib.md5(url.encode("utf-8"), usedforsecurity=False).hexdigest()

    def _format_for_upsert(
        self, article: dict[str, Any], category: str
    ) -> dict[str, Any] | None:
        """
        Transform a raw NewsAPI article dict into the structure expected by
        :meth:`~database_manager.DatabaseManager.upsert_vectors`.

        The returned dict has an ``embed_text`` key that the embedding engine
        uses.  That key must be removed before calling ``upsert_vectors`` —
        this is handled by :meth:`fetch_articles` and :meth:`fetch_all_topics`.

        Parameters
        ----------
        article:
            A single element from the ``articles`` array in the NewsAPI response.
        category:
            The topic/search query that produced this article (e.g. ``"AI"``).

        Returns
        -------
        dict or None
            Formatted article dict, or ``None`` if the article has no usable
            text (skipped silently).
        """
        title       = self._clean_text(article.get("title"))
        description = self._clean_text(article.get("description"))
        content     = self._clean_text(article.get("content"))
        url         = (article.get("url") or "").strip()
        published_at = article.get("publishedAt", "")
        source_name  = (article.get("source") or {}).get("name", "")
        author       = self._clean_text(article.get("author"))

        # Articles with neither title nor description carry no semantic value
        if not title and not description:
            logger.debug("Skipping article with no text content (url=%s).", url)
            return None

        # Skip NewsAPI's "[Removed]" placeholder articles
        if title == "[Removed]":
            return None

        if not url:
            logger.debug("Skipping article with no URL.")
            return None

        # ── embed_text: title + description for richer semantic coverage ──────
        parts = [p for p in (title, description) if p]
        embed_text = ". ".join(parts)

        # ── meta: full structured payload (stored verbatim in Endee as UTF-8) ─
        # The C++ backend stores this string as std::vector<uint8_t> bytes
        # via: vec.meta.assign(meta_str.begin(), meta_str.end())
        # On retrieval it is decoded from bytes → UTF-8 → JSON dict.
        meta = {
            "title":        title,
            "url":          url,
            "source":       source_name,
            "category":     category,
            "description":  description,
            "content":      content,
            "published_at": published_at,
            "author":       author,
        }

        # ── filter: flat categorical fields indexed by Endee's MDBX engine ────
        # Keep values lowercase to ensure consistent $eq comparisons.
        filter_data = {
            "source":   source_name.lower(),
            "category": category.lower(),
        }

        return {
            "id":           self._stable_id(url),
            "embed_text":   embed_text,
            "title":        meta["title"],
            "url":          meta["url"],
            # source / category: lowercase so Endee's $eq filter comparisons are
            # case-insensitive by convention (consistent with filter_data values).
            "source":       filter_data["source"],    # e.g. "techcrunch"
            "category":     filter_data["category"],  # e.g. "ai"
            "description":  meta["description"],
            "content":      meta["content"],
            "published_at": meta["published_at"],
            "author":       meta["author"],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_articles(
        self,
        topic: str,
        *,
        language: str = "en",
        sort_by: str = "publishedAt",
    ) -> list[dict[str, Any]]:
        """
        Fetch and format articles for a single *topic* from NewsAPI.

        Parameters
        ----------
        topic:
            The search query (e.g. ``"AI"``, ``"climate change"``).
        language:
            ISO 639-1 language code for the response articles (default: ``"en"``).
        sort_by:
            Sorting criterion.  One of ``"publishedAt"``, ``"relevancy"``,
            ``"popularity"`` (default: ``"publishedAt"``).

        Returns
        -------
        list[dict]
            Cleaned and formatted article dicts ready for embedding and upsert.
            Articles that fail quality checks are silently dropped.

        Raises
        ------
        RuntimeError
            If the NewsAPI HTTP request fails or returns a non-2xx status.
        """
        params = {
            "q":        topic,
            "language": language,
            "sortBy":   sort_by,
            "pageSize": self._cfg.page_size,
        }

        logger.info("Fetching articles for topic '%s' …", topic)

        try:
            resp = self._session.get(_NEWSAPI_ENDPOINT, params=params, timeout=15)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"NewsAPI request failed for topic '{topic}': {exc}") from exc

        if not resp.ok:
            raise RuntimeError(
                f"NewsAPI returned HTTP {resp.status_code} for topic '{topic}': {resp.text}"
            )

        data = resp.json()
        raw_articles: list[dict] = data.get("articles", [])
        logger.info(
            "Received %d raw article(s) for topic '%s'.",
            len(raw_articles),
            topic,
        )

        formatted: list[dict] = []
        for raw in raw_articles:
            result = self._format_for_upsert(raw, category=topic)
            if result is not None:
                formatted.append(result)

        logger.info(
            "Kept %d/%d article(s) after quality filtering for topic '%s'.",
            len(formatted),
            len(raw_articles),
            topic,
        )
        return formatted

    def fetch_all_topics(self) -> list[dict[str, Any]]:
        """
        Fetch articles for every topic defined in :attr:`~config.NewsAPIConfig.topics`.

        Duplicate articles (same URL, therefore same MD5 ID) that appear across
        multiple topic queries are deduplicated — the first occurrence wins.

        Returns
        -------
        list[dict]
            Deduplicated, merged list of formatted article dicts across all topics.
        """
        all_articles: list[dict] = []
        failed_topics: list[str] = []

        for topic in self._cfg.topics:
            try:
                articles = self.fetch_articles(topic)
                all_articles.extend(articles)
            except RuntimeError as exc:
                logger.error("Skipping topic '%s' due to error: %s", topic, exc)
                failed_topics.append(topic)

        # Deduplicate by vector ID (MD5 of URL)
        seen: set[str] = set()
        unique: list[dict] = []
        for article in all_articles:
            if article["id"] not in seen:
                seen.add(article["id"])
                unique.append(article)

        duplicate_count = len(all_articles) - len(unique)
        logger.info(
            "Fetched %d unique article(s) across %d topic(s). "
            "Dropped %d duplicate(s). Failed topic(s): %s",
            len(unique),
            len(self._cfg.topics) - len(failed_topics),
            duplicate_count,
            failed_topics or "none",
        )
        return unique
