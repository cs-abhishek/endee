"""
embeddings/manager.py — Sentence-embedding engine for Global News Pulse.

Wraps the ``sentence-transformers`` library to provide a clean, batched
text-to-vector interface.  The model (default: ``all-MiniLM-L6-v2``) is
loaded exactly once at construction time and stays in memory for the
lifetime of the process.

Design decisions
----------------
* **Normalised embeddings** — ``normalize_embeddings=True`` is passed to
  ``model.encode()``.  This projects every output vector onto the unit
  hypersphere, which makes cosine similarity equivalent to a dot product and
  matches the ``space_type: cosine`` setting in the Endee index config.
* **Lazy device selection** — CUDA is used when available; the CPU is the
  safe fallback.  No manual device management is needed.
* **Warmup on construction** — a single dummy forward pass warms up the
  model weights in CPU/GPU cache before real data arrives, keeping the first
  real batch fast.
"""

from __future__ import annotations

import logging
from typing import Sequence

from sentence_transformers import SentenceTransformer

from config import embedding_cfg, EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Transforms raw text into fixed-length float vectors via a pre-trained
    Sentence-BERT model.

    Parameters
    ----------
    config:
        An :class:`~config.EmbeddingConfig` instance.  Defaults to the
        module-level singleton read from environment variables.

    Attributes
    ----------
    dimension : int
        Embedding dimension reported by the loaded model (384 for
        ``all-MiniLM-L6-v2``).

    Examples
    --------
    >>> embedder = EmbeddingManager()
    >>> vectors = embedder.encode(["AI is changing the world", "Climate crisis deepens"])
    >>> len(vectors[0])
    384
    """

    def __init__(self, config: EmbeddingConfig = embedding_cfg) -> None:
        self._cfg = config
        self._model: SentenceTransformer = self._load_model()
        self._warmup()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_model(self) -> SentenceTransformer:
        """
        Download (first run) or load (cached) the sentence-transformer model.

        The model is stored in ``~/.cache/torch/sentence_transformers/`` by
        default.  Subsequent runs are instant.

        Returns
        -------
        SentenceTransformer
            The loaded, ready-to-use model instance.
        """
        logger.info("Loading sentence-transformer model '%s' …", self._cfg.model_name)
        model = SentenceTransformer(self._cfg.model_name)
        logger.info(
            "Model loaded — embedding dimension: %d",
            model.get_sentence_embedding_dimension(),
        )
        return model

    def _warmup(self) -> None:
        """
        Run a silent forward pass so the first real batch has no cold-start overhead.
        """
        logger.debug("Warming up model with dummy forward pass …")
        self._model.encode(["warmup"], show_progress_bar=False)
        logger.debug("Warmup complete.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding dimension of the loaded model."""
        return self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool | None = None,
    ) -> list[list[float]]:
        """
        Encode a list of strings into L2-normalised float vectors.

        Text is truncated to the model's maximum token length (256 tokens for
        ``all-MiniLM-L6-v2``) automatically — no manual truncation needed.

        Parameters
        ----------
        texts:
            Sequence of raw text strings to embed.
        show_progress:
            Whether to display a tqdm progress bar.  Defaults to ``True``
            when the batch size exceeds 50 items, ``False`` otherwise.

        Returns
        -------
        list[list[float]]
            A list of 384-dimensional float vectors, one per input string,
            in the same order as *texts*.

        Raises
        ------
        ValueError
            If *texts* is empty.
        """
        if not texts:
            raise ValueError("embed(): 'texts' sequence must not be empty.")

        # Default: show progress bar only for large batches (reduce terminal noise)
        display_bar = show_progress if show_progress is not None else len(texts) > 50

        logger.info(
            "Encoding %d text(s) with batch_size=%d …",
            len(texts),
            self._cfg.batch_size,
        )

        embeddings = self._model.encode(
            list(texts),
            batch_size=self._cfg.batch_size,
            show_progress_bar=display_bar,
            normalize_embeddings=True,   # ensures cosine sim == dot product
            convert_to_numpy=True,
        )

        vectors = embeddings.tolist()
        logger.info("Encoding complete — produced %d vector(s).", len(vectors))
        return vectors

    def encode_single(self, text: str) -> list[float]:
        """
        Convenience wrapper to encode exactly one string.

        Parameters
        ----------
        text:
            The string to embed.

        Returns
        -------
        list[float]
            A single 384-dimensional normalised float vector.
        """
        return self.encode([text], show_progress=False)[0]

    def encode_articles(
        self, articles: list[dict], text_key: str = "embed_text"
    ) -> list[list[float]]:
        """
        Encode a list of article dicts, reading the text from *text_key*.

        This is a convenience wrapper that pairs naturally with
        :meth:`~ingestion.news_provider.NewsProvider.fetch_all_topics`.

        Parameters
        ----------
        articles:
            List of article dicts; each must contain *text_key*.
        text_key:
            Dict key whose value is the text to embed.  Defaults to
            ``"embed_text"`` (the field written by ``NewsProvider``).

        Returns
        -------
        list[list[float]]
            Vectors in the same order as *articles*.

        Raises
        ------
        KeyError
            If any article dict is missing *text_key*.
        """
        texts = [a[text_key] for a in articles]
        return self.encode(texts)
