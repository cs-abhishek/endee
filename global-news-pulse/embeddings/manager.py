"""
embeddings/manager.py — Sentence-embedding engine for Global News Pulse.

Uses ``fastembed`` (ONNX runtime backend) instead of sentence-transformers /
PyTorch.  This eliminates the multi-gigabyte torch dependency and works
reliably on all platforms, including Streamlit Cloud.

The default model is ``BAAI/bge-small-en-v1.5``:
  - 384-dimensional output, identical to all-MiniLM-L6-v2
  - L2-normalised by default (cosine sim == dot product)
  - ONNX weights cached in the OS temp dir on first run (~23 MB)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from fastembed import TextEmbedding

from config import embedding_cfg, EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Transforms raw text into fixed-length float vectors via a pre-trained
    ONNX embedding model.

    Parameters
    ----------
    config:
        An :class:`~config.EmbeddingConfig` instance.  Defaults to the
        module-level singleton read from environment variables.

    Attributes
    ----------
    dimension : int
        Embedding dimension of the loaded model (384 for bge-small-en-v1.5).

    Examples
    --------
    >>> embedder = EmbeddingManager()
    >>> vectors = embedder.encode(["AI is changing the world", "Climate crisis deepens"])
    >>> len(vectors[0])
    384
    """

    def __init__(self, config: EmbeddingConfig = embedding_cfg) -> None:
        self._cfg = config
        self._model: TextEmbedding = self._load_model()
        self._dim: int = self._detect_dimension()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_model(self) -> TextEmbedding:
        logger.info("Loading fastembed model '%s' …", self._cfg.model_name)
        model = TextEmbedding(self._cfg.model_name)
        logger.info("fastembed model loaded.")
        return model

    def _detect_dimension(self) -> int:
        """Embed one dummy string to determine the output dimension."""
        dummy = list(self._model.embed(["warmup"]))
        dim = len(dummy[0])
        logger.info("Embedding dimension: %d", dim)
        return dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Embedding dimension of the loaded model."""
        return self._dim

    def encode(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool | None = None,  # kept for API compatibility
    ) -> list[list[float]]:
        """
        Encode a list of strings into L2-normalised float vectors.

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

        logger.info("Encoding %d text(s) …", len(texts))
        embeddings = list(self._model.embed(list(texts)))
        vectors = [e.tolist() for e in embeddings]
        logger.info("Encoding complete — produced %d vector(s).", len(vectors))
        return vectors

    def encode_single(self, text: str) -> list[float]:
        """Convenience wrapper to encode exactly one string."""
        return self.encode([text])[0]

    def encode_articles(
        self, articles: list[dict], text_key: str = "embed_text"
    ) -> list[list[float]]:
        """Encode a list of article dicts, reading the text from *text_key*."""
        texts = [a[text_key] for a in articles]
        return self.encode(texts)
