#!/usr/bin/env python3
"""
main_ingest.py — CLI entry point for the Global News Pulse ingestion pipeline.

This script orchestrates the full data pipeline in a single command:
    Fetch news → Embed text → Upsert to Endee

Usage
-----
    # Ingest all configured topics (default)
    python main_ingest.py

    # Ingest a single topic
    python main_ingest.py --topic "artificial intelligence"

    # Drop and rebuild the index before ingesting
    python main_ingest.py --recreate

    # Fetch & embed but do NOT write to Endee (useful for testing)
    python main_ingest.py --dry-run

    # Control log verbosity
    python main_ingest.py --log-level DEBUG

Pre-flight sequence
-------------------
Before any work begins, the script performs a mandatory pre-flight check:
    1. Verify the Endee server is reachable at ENDEE_HOST (default: localhost:8080).
    2. Verify the NEWS_API_KEY environment variable is set.
    3. Confirm or create the ``news_embeddings`` index with the correct schema.

If the Endee server is not running, the script prints clear instructions and exits.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Any

# ── Load .env BEFORE importing any local config modules ──────────────────────
# python-dotenv silently ignores a missing .env file, so this is always safe.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars may be set externally

from config import endee_cfg, news_cfg, embedding_cfg
from database_manager import DatabaseManager, EndeeConnectionError, EndeeIndexError
from embeddings.manager import EmbeddingManager
from ingestion.news_provider import NewsProvider

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _configure_logging(level: str) -> None:
    """Configure the root logger with a timestamped console handler."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

logger = logging.getLogger("global-news-pulse.ingest")

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

_ENDEE_DOCKER_HINT = """
  ┌─────────────────────────────────────────────────────────────┐
  │  Endee server is not running.  Start it with:              │
  │                                                             │
  │    docker compose up -d          (if using docker-compose) │
  │    — or —                                                   │
  │    docker run -p 8080:8080 endeeio/endee-server:latest     │
  │                                                             │
  │  Then re-run this script.                                   │
  └─────────────────────────────────────────────────────────────┘
"""


def preflight_check(db: DatabaseManager) -> bool:
    """
    Verify that the Endee server is alive and the NewsAPI key is configured.

    This function is intentionally verbose — its output is the first thing a
    recruiter (or new developer) sees when they run the project.

    Parameters
    ----------
    db:
        An initialised :class:`~database_manager.DatabaseManager` instance.

    Returns
    -------
    bool
        ``True`` if all checks pass; ``False`` otherwise (does not raise so
        the caller can decide whether to abort or continue).
    """
    all_ok = True

    # ── Check 1: Endee server health ─────────────────────────────────────────
    print("\n  [PRE-FLIGHT 1/2] Checking Endee server …", flush=True)
    try:
        health = db.health_check()
        print(
            f"  ✓  Endee is RUNNING  "
            f"(status={health.get('status')}, "
            f"host={endee_cfg.host})",
            flush=True,
        )
    except EndeeConnectionError:
        print(f"  ✗  Endee is NOT reachable at {endee_cfg.host}", flush=True)
        print(_ENDEE_DOCKER_HINT, flush=True)
        all_ok = False

    # ── Check 2: NewsAPI key ──────────────────────────────────────────────────
    print("\n  [PRE-FLIGHT 2/2] Checking NewsAPI key …", flush=True)
    if news_cfg.api_key:
        masked = news_cfg.api_key[:4] + "****" + news_cfg.api_key[-4:]
        print(f"  ✓  NEWS_API_KEY is set  ({masked})", flush=True)
    else:
        print(
            "  ✗  NEWS_API_KEY is not set.\n"
            "     Copy .env.example → .env and add your key from https://newsapi.org",
            flush=True,
        )
        all_ok = False

    print()
    return all_ok


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_ingestion(
    *,
    topic: str | None,
    recreate: bool,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    """
    Execute the full ingestion pipeline and return a summary dict.

    Pipeline stages
    ---------------
    1. **Pre-flight** — verify Endee health and API key.
    2. **Ensure collection** — create or verify the ``news_embeddings`` index.
    3. **Fetch articles** — pull from NewsAPI for the requested topic(s).
    4. **Embed** — batch-encode article text with sentence-transformers.
    5. **Upsert** — send vectors + metadata to Endee in configurable batches.

    Parameters
    ----------
    topic:
        Single topic to ingest.  ``None`` ingests all configured topics.
    recreate:
        If ``True``, drop the existing index before creating a new one.
    dry_run:
        If ``True``, skip the upsert step (no writes to Endee).
    batch_size:
        Number of vectors to upsert per HTTP request (reduces memory pressure
        when ingesting large article sets).

    Returns
    -------
    dict
        Summary with keys: ``articles_fetched``, ``articles_ingested``,
        ``elapsed_seconds``, ``dry_run``.
    """
    t_start = time.monotonic()
    summary: dict[str, Any] = {
        "articles_fetched":  0,
        "articles_ingested": 0,
        "elapsed_seconds":   0.0,
        "dry_run":           dry_run,
    }

    # ── Stage 1: Pre-flight ───────────────────────────────────────────────────
    db = DatabaseManager()
    print("=" * 66)
    print("  Global News Pulse — Ingestion Pipeline")
    print("=" * 66)

    if not preflight_check(db):
        logger.error("Pre-flight checks failed.  Aborting.")
        sys.exit(1)

    # ── Stage 2: Ensure Endee collection ─────────────────────────────────────
    logger.info("Stage 2/5 — Ensuring Endee collection …")
    try:
        db.ensure_collection(recreate=recreate)
    except EndeeIndexError as exc:
        logger.error("Failed to create/verify collection: %s", exc)
        sys.exit(1)

    # ── Stage 3: Fetch articles ───────────────────────────────────────────────
    logger.info("Stage 3/5 — Fetching news articles …")
    provider = NewsProvider()

    if topic:
        articles = provider.fetch_articles(topic)
    else:
        articles = provider.fetch_all_topics()

    summary["articles_fetched"] = len(articles)

    if not articles:
        logger.warning("No articles retrieved — nothing to ingest.")
        summary["elapsed_seconds"] = round(time.monotonic() - t_start, 2)
        return summary

    # ── Stage 4: Embed ────────────────────────────────────────────────────────
    logger.info("Stage 4/5 — Encoding %d article(s) …", len(articles))
    embedder = EmbeddingManager()
    vectors = embedder.encode_articles(articles)

    # Attach each vector to its article dict
    for article, vector in zip(articles, vectors):
        article["vector"] = vector

    # Remove the ephemeral embed_text field — it is not stored in Endee.
    # (DatabaseManager.upsert_vectors does not use this field anyway, but
    # keeping it clean avoids confusion in dry-run debug output.)
    for article in articles:
        article.pop("embed_text", None)

    # ── Stage 5: Upsert ───────────────────────────────────────────────────────
    if dry_run:
        logger.info(
            "Stage 5/5 — DRY RUN: skipping upsert for %d article(s).",
            len(articles),
        )
        # Print a sample so the user can verify the payload looks correct
        if articles:
            import json as _json
            sample = {k: v for k, v in articles[0].items() if k != "vector"}
            sample["vector"] = f"[{articles[0]['vector'][0]:.4f}, … ({len(articles[0]['vector'])} dims)]"
            print("\n  Sample payload (first article):")
            print("  " + _json.dumps(sample, indent=4, ensure_ascii=False).replace("\n", "\n  "))
    else:
        logger.info("Stage 5/5 — Upserting %d article(s) in batches of %d …", len(articles), batch_size)
        ingested = 0
        for i in range(0, len(articles), batch_size):
            chunk = articles[i : i + batch_size]
            try:
                db.upsert_vectors(chunk)
                ingested += len(chunk)
                logger.info(
                    "  Upserted batch %d/%d (%d vectors total so far).",
                    i // batch_size + 1,
                    -(-len(articles) // batch_size),  # ceiling division
                    ingested,
                )
            except (EndeeIndexError, EndeeConnectionError) as exc:
                logger.error("Upsert failed for batch starting at index %d: %s", i, exc)
                break

        summary["articles_ingested"] = ingested

    # ── Done ──────────────────────────────────────────────────────────────────
    summary["elapsed_seconds"] = round(time.monotonic() - t_start, 2)

    print()
    print("=" * 66)
    print("  Pipeline complete!")
    print(f"  Articles fetched  : {summary['articles_fetched']}")
    if not dry_run:
        total_in_db = db.count_vectors()
        print(f"  Articles ingested : {summary['articles_ingested']}")
        print(f"  Total in Endee    : {total_in_db}")
    else:
        print("  Upsert skipped    : dry-run mode")
    print(f"  Elapsed           : {summary['elapsed_seconds']}s")
    print("=" * 66)
    print()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main_ingest",
        description="Global News Pulse — fetch, embed, and store news articles in Endee.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main_ingest.py
  python main_ingest.py --topic "climate change"
  python main_ingest.py --recreate
  python main_ingest.py --dry-run --log-level DEBUG
        """,
    )

    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        metavar="QUERY",
        help=(
            "Fetch articles for a single topic/search query.  "
            "If omitted, all topics in NEWS_TOPICS are ingested."
        ),
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=False,
        help=(
            "Drop the existing Endee index and create a fresh one before ingesting.  "
            "WARNING: all previously stored vectors are permanently deleted."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Fetch and embed articles but do NOT write them to Endee.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Number of vectors per upsert batch (default: 50).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        metavar="LEVEL",
        help="Logging verbosity (default: INFO).",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    if args.recreate and not args.dry_run:
        print(
            "\n  WARNING: --recreate will PERMANENTLY DELETE all vectors in "
            f"'{endee_cfg.index_name}'.\n"
            "  Press Ctrl+C within 5 seconds to abort …\n",
            flush=True,
        )
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n  Aborted by user.")
            sys.exit(0)

    run_ingestion(
        topic=args.topic,
        recreate=args.recreate,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
