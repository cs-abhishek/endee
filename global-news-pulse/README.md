# Global News Pulse

A real-time semantic news trend analyzer built on top of the [Endee](https://github.com/endee-io/endee) vector database. It fetches live articles from NewsAPI, encodes them with a sentence-transformer model, stores the embeddings in an Endee HNSW index, and exposes two search modes through a Streamlit dashboard: a fast semantic lookup and a multi-step agentic intelligence brief powered by Llama 3.3 70B via Groq.

Online hosted link- https://abhishekk.streamlit.app/

---

## How it works

```
NewsAPI  -->  Sentence-Transformer  -->  Endee (HNSW + MDBX)
                                              |
                                      Streamlit UI
                                              |
                              Simple Search / Agentic Brief
                                              |
                                    Groq (Llama 3.3 70B)
```

**Simple Search** embeds the query and runs an approximate nearest-neighbour lookup against Endee. No LLM call. Results return in under one second.

**Agentic Search (Deep Dive)** runs a five-step pipeline:

1. Initial retrieval - semantic search, top 10 results.
2. Diversity check - if one source dominates (below the diversity threshold), the search widens to top 30 and merges novel articles.
3. Sub-trend extraction - the LLM reads the headlines and identifies 2-4 specific sub-trends or emerging angles.
4. Secondary retrieval - a targeted Endee search is run for each sub-trend.
5. Synthesis - all evidence (title, source, URL, similarity score) is passed to the LLM, which produces a structured markdown intelligence brief with inline citations.

---

## Prerequisites

| Requirement             | Version                       |
| ----------------------- | ----------------------------- |
| Python                  | 3.10+                         |
| Docker + Docker Compose | any recent version            |
| NewsAPI key             | free tier at newsapi.org      |
| Groq API key            | free tier at console.groq.com |

The Endee server runs inside Docker and is started with a single `docker compose up` command from the repository root.

---

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/<your-username>/endee.git
cd endee
```

### 2. Start the Endee vector database

```bash
docker compose up -d
```

This starts the Endee HTTP server on `http://localhost:8080`. No additional configuration is required for a local development setup.

### 3. Configure environment variables

```bash
cd global-news-pulse
cp .env.example .env
```

Open `.env` and fill in your keys:

```
NEWS_API_KEY=<your newsapi key>
GROQ_API_KEY=<your groq key>
```

All other values have sensible defaults and do not need to change for a standard local setup. See the [Configuration](#configuration) section for the full variable reference.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

The first run will download the `all-MiniLM-L6-v2` sentence-transformer model (~90 MB). Subsequent runs use the cached version.

---

## Running the ingestion pipeline

The ingestion pipeline fetches articles from NewsAPI, embeds them, and upserts them into Endee.

```bash
# Ingest all configured topics (AI, technology, science, finance, climate by default)
python main_ingest.py

# Ingest a single topic
python main_ingest.py --topic "quantum computing"

# Drop and rebuild the index before ingesting
python main_ingest.py --recreate

# Dry run: fetch and embed but do not write to Endee
python main_ingest.py --dry-run

# Verbose logging
python main_ingest.py --log-level DEBUG
```

A pre-flight check runs before any work begins. It verifies that the Endee server is reachable and that `NEWS_API_KEY` is set. If the server is not running, the script prints clear instructions and exits without doing anything.

---

## Running the Streamlit UI

```bash
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501` in a browser.

- Type a topic in the search box.
- Enable **Deep Dive** for the full agentic brief, or leave it off for a fast semantic search.
- The sidebar shows the current Endee and LLM configuration.

---

## Project structure

```
global-news-pulse/
    app.py                   Streamlit dashboard
    main_ingest.py           CLI ingestion pipeline
    config.py                Centralised configuration (env vars -> dataclasses)
    database_manager.py      Endee REST + MessagePack client
    requirements.txt
    .env.example             Template for environment variables

    embeddings/
        manager.py           sentence-transformers wrapper (all-MiniLM-L6-v2)

    ingestion/
        news_provider.py     NewsAPI fetcher, text cleaner, article formatter

    search/
        agent.py             NewsAgent: simple_search and agentic_search
        llm_client.py        Groq / OpenAI wrapper with the analyst system prompt
```

---

## Configuration

All settings are read from environment variables. Copy `.env.example` to `.env` and override only what you need.

### Endee

| Variable            | Default                 | Description                         |
| ------------------- | ----------------------- | ----------------------------------- |
| `ENDEE_HOST`        | `http://localhost:8080` | URL of the Endee server             |
| `ENDEE_AUTH_TOKEN`  | _(empty)_               | Auth token if enabled on the server |
| `ENDEE_INDEX_NAME`  | `news_embeddings`       | Name of the vector index            |
| `ENDEE_DIM`         | `384`                   | Embedding dimension                 |
| `ENDEE_SPACE_TYPE`  | `cosine`                | Distance metric                     |
| `ENDEE_PRECISION`   | `int16`                 | Quantisation precision              |
| `ENDEE_M`           | `16`                    | HNSW M parameter                    |
| `ENDEE_EF_CON`      | `200`                   | HNSW ef_construction                |
| `ENDEE_TIMEOUT_SEC` | `30`                    | HTTP request timeout (seconds)      |

### NewsAPI

| Variable         | Default                                 | Description                      |
| ---------------- | --------------------------------------- | -------------------------------- |
| `NEWS_API_KEY`   | _(required)_                            | NewsAPI key                      |
| `NEWS_TOPICS`    | `AI,technology,science,finance,climate` | Comma-separated topics to ingest |
| `NEWS_PAGE_SIZE` | `100`                                   | Articles per topic per request   |

### Embeddings

| Variable               | Default            | Description                      |
| ---------------------- | ------------------ | -------------------------------- |
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2` | sentence-transformers model name |
| `EMBEDDING_BATCH_SIZE` | `64`               | Encoding batch size              |

### LLM

| Variable                  | Default                   | Description                                                |
| ------------------------- | ------------------------- | ---------------------------------------------------------- |
| `GROQ_API_KEY`            | _(required)_              | Groq (or OpenAI) API key                                   |
| `LLM_BACKEND`             | `groq`                    | Provider: `groq` or `openai`                               |
| `LLM_MODEL`               | `llama-3.3-70b-versatile` | Model identifier                                           |
| `LLM_TEMPERATURE`         | `0.2`                     | Sampling temperature                                       |
| `LLM_MAX_TOKENS`          | `2048`                    | Max tokens per completion                                  |
| `LLM_DIVERSITY_THRESHOLD` | `0.4`                     | Source diversity ratio below which the agent widens search |

---

## Technical notes

**Vector storage and retrieval.** Endee uses HNSW for approximate nearest-neighbour search and MDBX as the metadata store. Inserts are sent as JSON over HTTP. Search responses are returned as MessagePack binary for efficiency. Each article is stored with a metadata blob (title, URL, source, category, description, content, published date, author) and a flat filter document (source, category) for Endee's indexed filter fields.

**Embeddings.** The `all-MiniLM-L6-v2` model produces 384-dimensional vectors. Embeddings are L2-normalised before upsert so cosine similarity reduces to a dot product on the unit sphere. The embed text is `title + ". " + description`.

**Deduplication.** Each article is assigned a stable ID derived from the MD5 hash of its URL. Re-ingesting the same article is a no-op via Endee's upsert semantics.

**LLM backend.** The default backend is Groq with `llama-3.3-70b-versatile` (~500 tokens/second on the free tier). Switching to OpenAI requires only setting `LLM_BACKEND=openai` and replacing `GROQ_API_KEY` with an OpenAI key; no code changes are needed.

---

## License

See [LICENSE](../LICENSE) in the repository root.
