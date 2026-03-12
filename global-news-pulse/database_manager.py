"""
database_manager.py — Endee vector database interface for Global News Pulse.

This module is the single source of truth for all interactions with the Endee
vector database.  It wraps the Endee REST API (http://localhost:8080) and
exposes a clean, high-level interface that the rest of the application uses.

Architecture note
-----------------
Endee is a self-hosted, high-performance vector database written in C++ that
exposes a REST API.  Under the hood it runs an HNSW ANN index with optional
quantisation and a fast MDBX-backed metadata store.

The connection model is pure HTTP — there is no persistent socket or client
object.  Every public method opens a short-lived ``requests.Session`` call,
so this class is thread-safe and can be used from multiple workers.

Wire format
-----------
* Insert  : JSON  (``Content-Type: application/json``)
* Search  : POST  → response is **MessagePack** (``Content-Type: application/msgpack``)
  The server packs a ``ResultSet`` struct (an array whose single element is the
  list of ``VectorResult`` arrays) using the msgpack-c library.

Metadata storage strategy
--------------------------
Endee vectors carry two string fields:

``meta`` (bytes)  — arbitrary binary payload, stored verbatim.
    We serialize a JSON dict here: ``{"title": ..., "url": ..., "source": ...}``.

``filter`` (JSON string) — a flat JSON object whose keys are indexed for O(1)
    filtered ANN search at query time.
    We write ``{"source": "reuters", "category": "technology"}`` here so that
    callers can later pass ``[{"source": {"$eq": "reuters"}}]`` in search queries.

Usage
-----
>>> from database_manager import DatabaseManager
>>> db = DatabaseManager()
>>> db.ensure_collection()
>>> db.upsert_vectors([
...     {"id": "abc123", "vector": [0.1, ...], "title": "...", "url": "...", "source": "bbc"}
... ])
>>> results = db.similarity_search([0.1, ...], top_k=10)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import msgpack
import requests

from config import endee_cfg, EndeeConfig

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class EndeeConnectionError(RuntimeError):
    """Raised when the Endee server is unreachable."""


class EndeeIndexError(RuntimeError):
    """Raised for index-level errors (e.g. wrong dimension)."""


class EndeeSearchError(RuntimeError):
    """Raised when a similarity search fails."""


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """
    High-level interface for the Endee vector database.

    Parameters
    ----------
    config:
        An :class:`~config.EndeeConfig` instance.  Defaults to the
        module-level singleton read from environment variables.

    Examples
    --------
    >>> db = DatabaseManager()
    >>> db.ensure_collection()
    >>> db.upsert_vectors([...])
    >>> results = db.similarity_search(query_vector, top_k=5)
    """

    # msgpack position indices for a VectorResult array
    # MSGPACK_DEFINE order: similarity, id, meta, filter, norm, vector
    _RES_SIMILARITY = 0
    _RES_ID         = 1
    _RES_META       = 2
    _RES_FILTER     = 3
    _RES_NORM       = 4
    _RES_VECTOR     = 5

    def __init__(self, config: EndeeConfig = endee_cfg) -> None:
        self._cfg = config
        self._base_url = config.host.rstrip("/")
        self._session = self._build_session()
        logger.info(
            "DatabaseManager initialised → host=%s  index=%s  dim=%d",
            self._base_url,
            self._cfg.index_name,
            self._cfg.embedding_dim,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """
        Build a :class:`requests.Session` with auth and content-type headers
        pre-populated.

        Endee's auth model: if ``NDD_AUTH_TOKEN`` was set on the server, every
        request must include ``Authorization: <token>`` (no ``Bearer`` prefix).
        If the server runs in open mode (no token) the header is simply omitted.
        """
        session = requests.Session()
        if self._cfg.auth_token:
            session.headers["Authorization"] = self._cfg.auth_token
        session.headers["Content-Type"] = "application/json"
        return session

    def _url(self, path: str) -> str:
        """Construct a full API URL from a relative *path*."""
        return f"{self._base_url}{path}"

    def _raise_for_status(self, resp: requests.Response, context: str) -> None:
        """
        Raise a descriptive exception for non-2xx responses.

        Parameters
        ----------
        resp:    The HTTP response object.
        context: A short description of the operation (for error messages).
        """
        if not resp.ok:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise EndeeIndexError(
                f"[{context}] HTTP {resp.status_code}: {detail}"
            )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, Any]:
        """
        Ping the Endee health endpoint.

        Returns
        -------
        dict
            Server status payload, e.g. ``{"status": "ok", "timestamp": ...}``.

        Raises
        ------
        EndeeConnectionError
            If the server is unreachable within the configured timeout.
        """
        try:
            resp = self._session.get(
                self._url("/api/v1/health"),
                timeout=self._cfg.request_timeout,
            )
            self._raise_for_status(resp, "health_check")
            data = resp.json()
            logger.debug("Health check OK: %s", data)
            return data
        except requests.exceptions.ConnectionError as exc:
            raise EndeeConnectionError(
                f"Cannot reach Endee at {self._base_url}. "
                "Is the server running? (docker compose up -d)"
            ) from exc

    # ------------------------------------------------------------------
    # Collection / index management
    # ------------------------------------------------------------------

    def ensure_collection(self, *, recreate: bool = False) -> None:
        """
        Create the ``news_embeddings`` collection if it does not exist.

        This is idempotent — calling it multiple times is safe.  Endee returns
        HTTP 409 when an index already exists; this method silently ignores that
        specific error.

        Parameters
        ----------
        recreate:
            If ``True``, delete the existing collection first and rebuild it
            from scratch.  **All stored vectors will be lost.**  Use with care.

        Raises
        ------
        EndeeConnectionError
            If the server is unreachable.
        EndeeIndexError
            For any unexpected server-side error.
        """
        if recreate:
            self._delete_collection_if_exists()

        payload = {
            "index_name":  self._cfg.index_name,
            "dim":         self._cfg.embedding_dim,
            "space_type":  self._cfg.space_type,
            "M":           self._cfg.m,
            "ef_con":      self._cfg.ef_construction,
            "precision":   self._cfg.precision,
        }

        logger.info(
            "Ensuring collection '%s' (dim=%d, space=%s, precision=%s) …",
            self._cfg.index_name,
            self._cfg.embedding_dim,
            self._cfg.space_type,
            self._cfg.precision,
        )

        try:
            resp = self._session.post(
                self._url("/api/v1/index/create"),
                json=payload,
                timeout=self._cfg.request_timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise EndeeConnectionError(
                f"Cannot reach Endee at {self._base_url}."
            ) from exc

        if resp.status_code == 409:
            logger.info("Collection '%s' already exists — skipping creation.", self._cfg.index_name)
            return

        self._raise_for_status(resp, "ensure_collection")
        logger.info("Collection '%s' created successfully.", self._cfg.index_name)

    def _delete_collection_if_exists(self) -> None:
        """Delete the collection, ignoring 404 if it does not exist."""
        try:
            resp = self._session.delete(
                self._url(f"/api/v1/index/{self._cfg.index_name}/delete"),
                timeout=self._cfg.request_timeout,
            )
            if resp.status_code not in (200, 404):
                self._raise_for_status(resp, "_delete_collection_if_exists")
            logger.warning("Collection '%s' deleted.", self._cfg.index_name)
        except requests.exceptions.ConnectionError as exc:
            raise EndeeConnectionError(
                f"Cannot reach Endee at {self._base_url}."
            ) from exc

    def get_collection_info(self) -> dict[str, Any]:
        """
        Return metadata about the ``news_embeddings`` index.

        Returns
        -------
        dict
            Keys include ``total_elements``, ``dimension``, ``space_type``,
            ``precision``, ``M``, ``ef_con``.

        Raises
        ------
        EndeeIndexError
            If the collection does not exist or the request fails.
        """
        try:
            resp = self._session.get(
                self._url(f"/api/v1/index/{self._cfg.index_name}/info"),
                timeout=self._cfg.request_timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise EndeeConnectionError(str(exc)) from exc

        self._raise_for_status(resp, "get_collection_info")
        return resp.json()

    # ------------------------------------------------------------------
    # Vector ingestion
    # ------------------------------------------------------------------

    def upsert_vectors(self, items: list[dict[str, Any]]) -> int:
        """
        Insert or overwrite a batch of vectors with metadata.

        Endee supports overwriting by vector ID — re-inserting with the same
        ``id`` replaces the stored vector and metadata.

        Parameters
        ----------
        items:
            A list of dicts, each containing:

            * ``vector``  (list[float], required) — the embedding.
            * ``id``      (str, optional) — stable ID; a UUID4 is generated if omitted.
            * ``title``   (str, optional) — article headline.
            * ``url``     (str, optional) — article URL.
            * ``source``  (str, optional) — news outlet name.
            * ``category``(str, optional) — topic / category string.
            * ``published_at`` (str, optional) — ISO-8601 timestamp.

        Returns
        -------
        int
            Number of vectors submitted.

        Raises
        ------
        EndeeIndexError
            If the server rejects the batch (e.g. wrong dimension).
        ValueError
            If *items* is empty or a required field is missing.

        Notes
        -----
        The metadata dict is stored verbatim in the ``meta`` field as
        UTF-8-encoded JSON bytes.  The ``filter`` field receives a flat JSON
        object containing ``source`` and ``category`` so that Endee can index
        these fields for fast filtered ANN search.
        """
        if not items:
            raise ValueError("upsert_vectors: 'items' list must not be empty.")

        batch = []
        for item in items:
            vector = item.get("vector")
            if not vector:
                raise ValueError(f"Item missing required 'vector' field: {item}")

            if len(vector) != self._cfg.embedding_dim:
                raise EndeeIndexError(
                    f"Vector dimension mismatch: expected {self._cfg.embedding_dim}, "
                    f"got {len(vector)}."
                )

            vector_id = str(item.get("id") or uuid.uuid4())

            # ── meta (rich payload stored as UTF-8 JSON bytes) ──────────────
            # The C++ server receives this as a string and stores each byte of the
            # UTF-8 encoding into std::vector<uint8_t> via
            #   vec.meta.assign(meta_str.begin(), meta_str.end())
            # On retrieval via msgpack the bytes are decoded → UTF-8 → JSON dict.
            meta_payload = {
                "title":        item.get("title", ""),
                "url":          item.get("url", ""),
                "source":       item.get("source", ""),
                "category":     item.get("category", ""),
                "published_at": item.get("published_at", ""),
                "description":  item.get("description", ""),
                "content":      item.get("content", ""),
                "author":       item.get("author", ""),
            }

            # ── filter (flat JSON object for Endee's indexed filter fields) ──
            # Endee infers field types on first insertion.  Keep these to
            # categorical/string values to enable $eq / $in queries at search time.
            filter_payload = {
                "source":   item.get("source", ""),
                "category": item.get("category", ""),
            }

            batch.append({
                "id":     vector_id,
                "vector": [float(v) for v in vector],
                # meta must be a string per the Endee JSON insert schema
                "meta":   json.dumps(meta_payload, ensure_ascii=False),
                "filter": json.dumps(filter_payload, ensure_ascii=False),
            })

        logger.info(
            "Upserting %d vector(s) into '%s' …",
            len(batch),
            self._cfg.index_name,
        )

        try:
            resp = self._session.post(
                self._url(f"/api/v1/index/{self._cfg.index_name}/vector/insert"),
                data=json.dumps(batch),
                timeout=self._cfg.request_timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise EndeeConnectionError(str(exc)) from exc

        self._raise_for_status(resp, "upsert_vectors")
        logger.info("Successfully upserted %d vector(s).", len(batch))
        return len(batch)

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        *,
        source_filter: str | None = None,
        category_filter: str | None = None,
        ef: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Run an approximate nearest-neighbour search against the news index.

        Parameters
        ----------
        query_vector:
            The query embedding (must be 384-dimensional for all-MiniLM-L6-v2).
        top_k:
            Number of nearest neighbours to retrieve (1–4096).
        source_filter:
            Optionally restrict results to vectors whose ``source`` field
            equals this value.  Uses Endee's ``$eq`` operator.
        category_filter:
            Optionally restrict results to a specific category.
        ef:
            Runtime HNSW search depth.  ``0`` means use the server default
            (``ef_construction``).  Higher values improve recall at the cost
            of latency.

        Returns
        -------
        list[dict]
            Ranked list of results, most similar first.  Each dict contains:

            * ``id``         — vector ID
            * ``similarity`` — cosine similarity (0–1; higher = more similar)
            * ``title``      — article headline
            * ``url``        — article URL
            * ``source``     — news outlet
            * ``category``   — topic
            * ``published_at`` — ISO-8601 timestamp
            * ``description`` — article snippet

        Raises
        ------
        EndeeSearchError
            If the server returns an error or the response cannot be decoded.
        ValueError
            If *query_vector* has the wrong dimension.
        """
        if len(query_vector) != self._cfg.embedding_dim:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self._cfg.embedding_dim}, "
                f"got {len(query_vector)}."
            )

        # Build optional filter expression (Endee array-based filter format)
        filter_clauses: list[dict] = []
        if source_filter:
            filter_clauses.append({"source": {"$eq": source_filter}})
        if category_filter:
            filter_clauses.append({"category": {"$eq": category_filter}})

        payload: dict[str, Any] = {
            "vector": [float(v) for v in query_vector],
            "k":      top_k,
        }
        if ef > 0:
            payload["ef"] = ef
        if filter_clauses:
            # Endee expects the filter as a JSON-encoded string, not a raw object
            payload["filter"] = json.dumps(filter_clauses)

        logger.debug(
            "Searching '%s': top_k=%d, filter=%s",
            self._cfg.index_name,
            top_k,
            filter_clauses or None,
        )

        try:
            # Override Content-Type for this request — search endpoint returns msgpack
            resp = self._session.post(
                self._url(f"/api/v1/index/{self._cfg.index_name}/search"),
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self._cfg.request_timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise EndeeConnectionError(str(exc)) from exc

        if not resp.ok:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise EndeeSearchError(
                f"Search failed — HTTP {resp.status_code}: {detail}"
            )

        # ── Decode the msgpack response ──────────────────────────────────────
        # Wire layout (msgpack-c MSGPACK_DEFINE order):
        #   ResultSet              →  [ results_array ]
        #   results_array          →  [ VectorResult, … ]
        #   VectorResult           →  [ similarity, id, meta_bytes, filter_str, norm, vector ]
        try:
            raw = msgpack.unpackb(resp.content, raw=False, strict_map_key=False)
        except Exception as exc:
            raise EndeeSearchError(
                f"Failed to decode msgpack search response: {exc}"
            ) from exc

        # raw is a 1-element list: [ [result, result, ...] ]
        if not raw or not isinstance(raw, list):
            return []

        results_array = raw[0] if isinstance(raw[0], list) else raw
        results: list[dict[str, Any]] = []

        for entry in results_array:
            try:
                similarity  = float(entry[self._RES_SIMILARITY])
                vector_id   = str(entry[self._RES_ID])
                meta_bytes  = entry[self._RES_META]
                filter_str  = entry[self._RES_FILTER]

                # meta_bytes is a bytes/bytearray from raw=False with BIN type
                if isinstance(meta_bytes, (bytes, bytearray)):
                    meta_dict = json.loads(meta_bytes.decode("utf-8"))
                elif isinstance(meta_bytes, str):
                    meta_dict = json.loads(meta_bytes)
                else:
                    meta_dict = {}

                results.append({
                    "id":           vector_id,
                    "similarity":   round(similarity, 6),
                    "title":        meta_dict.get("title", ""),
                    "url":          meta_dict.get("url", ""),
                    "source":       meta_dict.get("source", ""),
                    "category":     meta_dict.get("category", ""),
                    "published_at": meta_dict.get("published_at", ""),
                    "description":  meta_dict.get("description", ""),
                    "content":      meta_dict.get("content", ""),
                    "author":       meta_dict.get("author", ""),
                    "filter_data":  filter_str,
                })
            except (IndexError, KeyError, json.JSONDecodeError) as exc:
                logger.warning("Skipping malformed search result: %s — %s", entry, exc)
                continue

        logger.info(
            "Search returned %d result(s) from '%s'.",
            len(results),
            self._cfg.index_name,
        )
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_collections(self) -> list[dict[str, Any]]:
        """
        Return all indexes visible to the current user.

        Returns
        -------
        list[dict]
            Each dict contains ``name``, ``dimension``, ``space_type``,
            ``precision``, ``total_elements``, ``created_at``.
        """
        try:
            resp = self._session.get(
                self._url("/api/v1/index/list"),
                timeout=self._cfg.request_timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise EndeeConnectionError(str(exc)) from exc

        self._raise_for_status(resp, "list_collections")
        return resp.json().get("indexes", [])

    def count_vectors(self) -> int:
        """Return the number of vectors currently stored in the news index."""
        info = self.get_collection_info()
        return int(info.get("total_elements", 0))

    def wait_for_server(self, retries: int = 10, delay: float = 2.0) -> None:
        """
        Block until the Endee server becomes healthy or *retries* is exhausted.

        Useful in Docker Compose start-up scripts that launch the Python
        application alongside the Endee container.

        Parameters
        ----------
        retries:
            Maximum number of health-check attempts.
        delay:
            Seconds to wait between attempts.

        Raises
        ------
        EndeeConnectionError
            If the server is still unreachable after all retries.
        """
        logger.info("Waiting for Endee server at %s …", self._base_url)
        for attempt in range(1, retries + 1):
            try:
                self.health_check()
                logger.info("Endee server is ready (attempt %d/%d).", attempt, retries)
                return
            except EndeeConnectionError:
                logger.warning(
                    "Attempt %d/%d — server not ready, retrying in %.1fs …",
                    attempt,
                    retries,
                    delay,
                )
                time.sleep(delay)

        raise EndeeConnectionError(
            f"Endee server at {self._base_url} did not become ready after "
            f"{retries} attempts.  Check that the container is running:\n"
            "    docker compose up -d"
        )
