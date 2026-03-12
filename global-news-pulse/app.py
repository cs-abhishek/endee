"""
app.py — Streamlit dashboard for Global News Pulse.

Run with:
    streamlit run app.py

The UI offers two modes:
  - Simple Search  : fast semantic search, returns top-5 ranked articles.
  - Deep Dive      : full agentic pipeline — sub-trend extraction, diversity
                     check, secondary searches, and a structured LLM brief.
"""

from __future__ import annotations

# ── Load .env before any local imports that read os.getenv ───────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import time
import streamlit as st

from config import endee_cfg, llm_cfg

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Global News Pulse",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean card styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Headline card */
    .result-card {
        border: 1px solid #2d2d2d;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        background: #1a1a2e;
    }
    .result-card h4 { margin: 0 0 6px 0; color: #e0e0e0; font-size: 1rem; }
    .result-card .meta { font-size: 0.78rem; color: #888; margin-bottom: 8px; }
    .result-card .description { font-size: 0.88rem; color: #bbb; }
    .similarity-bar { height: 6px; border-radius: 3px; background: #0f3460;
                       margin-bottom: 8px; }
    .similarity-fill { height: 100%; border-radius: 3px; background: #e94560; }
    /* Source badge */
    .source-badge {
        display: inline-block;
        background: #0f3460;
        color: #a8d8ea;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin-right: 6px;
    }
    /* Section header */
    .section-header {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #e94560;
        margin: 20px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resource singletons — loaded once per Streamlit session
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model …")
def get_agent():
    """
    Instantiate :class:`~search.agent.NewsAgent` once per Streamlit process.

    Heavy resources (sentence-transformer model, HTTP sessions) are only
    created once and reused across all user interactions.
    """
    from search.agent import NewsAgent
    return NewsAgent()


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------


def _similarity_bar(score: float) -> str:
    """Return an HTML similarity progress bar for *score* (0–1)."""
    pct = int(score * 100)
    return (
        f'<div class="similarity-bar">'
        f'<div class="similarity-fill" style="width:{pct}%"></div></div>'
    )


def render_article_card(article: dict, rank: int) -> None:
    """Render a single news article as a styled card."""
    sim = article.get("similarity", 0.0)
    title = article.get("title") or "Untitled"
    source = article.get("source") or "Unknown source"
    url = article.get("url") or "#"
    description = article.get("description") or ""
    published = (article.get("published_at") or "")[:10]  # YYYY-MM-DD only
    category = article.get("category") or ""

    badge_html = f'<span class="source-badge">{source}</span>'
    if category:
        badge_html += f'<span class="source-badge">{category}</span>'

    st.markdown(
        f"""
        <div class="result-card">
          {_similarity_bar(sim)}
          <h4>#{rank} — <a href="{url}" target="_blank"
              style="color:#e0e0e0;text-decoration:none;">{title}</a></h4>
          <div class="meta">{badge_html} &nbsp; {published}</div>
          <div class="description">{description[:260]}{"…" if len(description) > 260 else ""}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simple_results(results: list[dict]) -> None:
    """Render a list of simple-search article cards."""
    if not results:
        st.info("No results found.  Try a different query or run the ingestion pipeline first.")
        return

    st.markdown('<p class="section-header">Top Results</p>', unsafe_allow_html=True)
    for i, article in enumerate(results, start=1):
        render_article_card(article, i)


def render_agentic_brief(brief) -> None:
    """Render the full agentic intelligence brief and citation index."""
    # ── Performance metrics ───────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Sources cited", len(brief.sources))
    col_b.metric("Sub-trends", len(brief.sub_trends))
    col_c.metric("Initial results", len(brief.initial_results))
    col_d.metric("Elapsed", f"{brief.elapsed_seconds}s")

    if brief.diversity_expanded:
        st.warning(
            "**Diversity Expansion triggered** — initial results were concentrated "
            "in a single source.  The agent automatically broadened the search to "
            "surface differing perspectives.",
            icon="🔄",
        )

    # ── Sub-trend chips ───────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Identified Sub-Trends</p>', unsafe_allow_html=True)
    chip_html = " &nbsp; ".join(
        f'<span class="source-badge" style="font-size:0.82rem;padding:4px 14px;">{t}</span>'
        for t in brief.sub_trends
    )
    st.markdown(chip_html, unsafe_allow_html=True)
    st.markdown("")

    # ── Intelligence Brief ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Intelligence Brief</p>', unsafe_allow_html=True)
    st.markdown(brief.summary)

    # ── Evidence — expandable raw results ────────────────────────────────────
    with st.expander(f"📰 Initial Evidence ({len(brief.initial_results)} articles)", expanded=False):
        for i, article in enumerate(brief.initial_results, start=1):
            render_article_card(article, i)

    for trend, articles in brief.secondary_evidence.items():
        if articles:
            with st.expander(f"🔍 Sub-trend: {trend} ({len(articles)} articles)", expanded=False):
                for i, article in enumerate(articles, start=1):
                    render_article_card(article, i)

    # ── Citation Index ────────────────────────────────────────────────────────
    with st.expander(f"📌 Citation Index ({len(brief.sources)} sources)", expanded=False):
        st.markdown('<p class="section-header">All Referenced Sources</p>', unsafe_allow_html=True)
        for i, src in enumerate(brief.sources, start=1):
            url = src.get("url", "#")
            title = src.get("title") or "Untitled"
            source = src.get("source") or "Unknown"
            pub = (src.get("published_at") or "")[:10]
            sim = src.get("similarity", 0.0)
            st.markdown(
                f"**[{i}]** [{title}]({url})  \n"
                f"<span class='source-badge'>{source}</span>"
                f"<span style='font-size:0.75rem;color:#888;'> {pub} · sim {sim:.3f}</span>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/endee-io/endee/master/docs/assets/logo.png",
            use_column_width=True,
        ) if False else None  # placeholder if logo not available

        st.title("🌐 Global News Pulse")
        st.caption("Powered by **Endee** vector DB + **Llama 3.3** via Groq")
        st.divider()

        st.subheader("Configuration")
        st.code(
            f"Endee host : {endee_cfg.host}\n"
            f"Index      : {endee_cfg.index_name}\n"
            f"Dimensions : {endee_cfg.embedding_dim}\n"
            f"LLM model  : {llm_cfg.model}\n"
            f"LLM backend: {llm_cfg.backend}",
            language=None,
        )
        st.divider()

        st.subheader("Quick start")
        st.markdown(
            """
            1. Start Endee: `docker compose up -d`
            2. Ingest news: `python main_ingest.py`
            3. Search here!
            """
        )
        st.divider()

        st.caption("Fork of [endee-io/endee](https://github.com/endee-io/endee)")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    render_sidebar()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:0;'>🌐 Global News Pulse</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#888;margin-top:4px;'>Real-time semantic trend analysis "
        "powered by <strong>Endee</strong> vector DB and <strong>Llama 3.3 70B</strong></p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Search controls ───────────────────────────────────────────────────────
    with st.form("search_form", clear_on_submit=False):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            query = st.text_input(
                label="Search query",
                placeholder="e.g.  AI regulation in the EU,  semiconductor supply chain,  climate finance …",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("🔍 Search", use_container_width=True)

        deep_dive = st.checkbox(
            "🧠 Deep Dive — Agentic Analysis  "
            "*(multi-step sub-trend extraction + LLM intelligence brief)*",
            value=False,
        )

    # ── Search execution ──────────────────────────────────────────────────────
    if submitted and query.strip():
        try:
            agent = get_agent()
        except Exception as exc:
            st.error(
                f"Could not initialise the search agent: {exc}\n\n"
                "Make sure GROQ_API_KEY and ENDEE_HOST are configured. "
                "On Streamlit Cloud add them under **Settings > Secrets**."
            )
            st.stop()

        if deep_dive:
            # ── Agentic mode ──────────────────────────────────────────────────
            st.markdown('<p class="section-header">Agentic Intelligence Pipeline</p>', unsafe_allow_html=True)

            # Live step tracker
            status_box = st.empty()
            steps = [
                "Encoding query …",
                "Searching Endee (initial retrieval) …",
                "Running source diversity check …",
                "Extracting sub-trends via LLM …",
                "Running secondary searches …",
                "Synthesising intelligence brief …",
            ]

            try:
                with st.spinner("Running agentic pipeline …"):
                    # Show step progress while the agent works
                    progress_bar = st.progress(0, text=steps[0])
                    for i, step_text in enumerate(steps):
                        progress_bar.progress(
                            int((i / len(steps)) * 100), text=f"Step {i+1}/{len(steps)}: {step_text}"
                        )
                        if i == 0:
                            # Kick off the actual work after first step shown
                            t0 = time.monotonic()
                            brief = agent.agentic_search(query.strip())
                            break

                    progress_bar.progress(100, text="Done ✓")

                status_box.empty()
                render_agentic_brief(brief)
            except Exception as exc:
                st.error(f"Agentic search failed: {exc}")

        else:
            # ── Simple search mode ────────────────────────────────────────────
            try:
                with st.spinner("Searching …"):
                    t0 = time.monotonic()
                    results = agent.simple_search(query.strip())
                    elapsed = round(time.monotonic() - t0, 3)

                st.caption(f"Returned {len(results)} result(s) in {elapsed}s")
                render_simple_results(results)
            except Exception as exc:
                st.error(f"Search failed: {exc}")

    elif submitted:
        st.warning("Please enter a search query.")

    # ── Empty state ───────────────────────────────────────────────────────────
    else:
        st.markdown(
            """
            <div style="text-align:center;padding:60px 0;color:#555;">
              <div style="font-size:3rem;">🌐</div>
              <h3 style="color:#666;">Enter a topic above to begin</h3>
              <p>Use <strong>Simple Search</strong> for fast semantic results<br>
              or enable <strong>Deep Dive</strong> for agentic multi-step analysis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
