"""
config.py — Centralised configuration for Global News Pulse.

All runtime settings are read from environment variables so the project
is fully Plug-and-Play: copy .env.example to .env, fill in your keys, and go.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EndeeConfig:
    """Connection settings for the Endee vector database server."""

    host: str = field(default_factory=lambda: os.getenv("ENDEE_HOST", "http://localhost:8080"))
    auth_token: str = field(default_factory=lambda: os.getenv("ENDEE_AUTH_TOKEN", ""))
    # all-MiniLM-L6-v2 produces 384-dimensional embeddings
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("ENDEE_DIM", "384")))
    index_name: str = field(default_factory=lambda: os.getenv("ENDEE_INDEX_NAME", "news_embeddings"))
    # HNSW build-time parameters
    m: int = field(default_factory=lambda: int(os.getenv("ENDEE_M", "16")))
    ef_construction: int = field(default_factory=lambda: int(os.getenv("ENDEE_EF_CON", "200")))
    # Quantisation — int16 gives a good accuracy/memory trade-off
    precision: str = field(default_factory=lambda: os.getenv("ENDEE_PRECISION", "int16"))
    # Space type: cosine is best suited for normalised sentence embeddings
    space_type: str = field(default_factory=lambda: os.getenv("ENDEE_SPACE_TYPE", "cosine"))
    request_timeout: int = field(default_factory=lambda: int(os.getenv("ENDEE_TIMEOUT_SEC", "30")))


@dataclass(frozen=True)
class NewsAPIConfig:
    """Configuration for the NewsAPI data source."""

    api_key: str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))
    page_size: int = field(default_factory=lambda: int(os.getenv("NEWS_PAGE_SIZE", "100")))
    # Comma-separated list of topics to track, e.g. "AI,climate,finance"
    topics: list = field(
        default_factory=lambda: os.getenv("NEWS_TOPICS", "AI,technology,science,finance,climate").split(",")
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the sentence-transformer encoder."""

    model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    batch_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "64")))


@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration for the LLM intelligence layer.

    Supports Groq (default, free-tier, ~500 tok/s) or OpenAI as backends.
    Set ``LLM_BACKEND=openai`` and supply ``GROQ_API_KEY`` with your OpenAI key
    to switch providers without changing any other code.
    """

    # API key: used for both Groq and OpenAI (env var name is shared for simplicity)
    api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    # LLM provider: 'groq' or 'openai'
    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "groq"))
    # Model identifier — for Groq: llama-3.3-70b-versatile, for OpenAI: gpt-4o-mini
    model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    )
    # Low temperature → deterministic analytical output
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )
    # Diversity threshold: if unique_sources / total_results falls below this,
    # the agent automatically expands the search for differing perspectives.
    diversity_threshold: float = field(
        default_factory=lambda: float(os.getenv("LLM_DIVERSITY_THRESHOLD", "0.4"))
    )


# Module-level singletons — import these directly in other modules
endee_cfg = EndeeConfig()
news_cfg = NewsAPIConfig()
embedding_cfg = EmbeddingConfig()
llm_cfg = LLMConfig()
