# ui/app.py
import streamlit as st
import requests

st.set_page_config(
    page_title="DevDocs AI",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,300&display=swap');

:root {
    --bg:         #0d1117;
    --bg2:        #161b22;
    --bg3:        #1c2333;
    --amber:      #f0a500;
    --teal:       #39d0c8;
    --green:      #3fb950;
    --red:        #f85149;
    --txt:        #e6edf3;
    --txt2:       #8b949e;
    --border:     #30363d;
}

/* ── Reset & base ── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
html, body, .stApp, .main, .block-container {
    background: var(--bg) !important;
    color: var(--txt) !important;
    font-family: 'Fraunces', Georgia, serif;
    padding: 0 !important;
    max-width: 100% !important;
}
.block-container { padding: 0 2rem 6rem 2rem !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 1.5rem 1.25rem !important; }
section[data-testid="stSidebar"] * { color: var(--txt) !important; }

/* ── ALL buttons — base reset ── */
.stButton button {
    all: unset !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    cursor: pointer !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    padding: 0.55rem 1rem !important;
    transition: opacity 0.15s !important;
    white-space: nowrap !important;
}

/* Primary ask button */
div[data-testid="stFormSubmitButton"] button {
    background: var(--amber) !important;
    color: #000 !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    padding: 0.6rem 1.2rem !important;
    height: 42px !important;
}
div[data-testid="stFormSubmitButton"] button:hover { opacity: 0.85 !important; }

/* Example question buttons */
.example-wrap .stButton button {
    background: var(--bg3) !important;
    color: var(--teal) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.7rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.5rem 0.85rem !important;
}
.example-wrap .stButton button:hover { border-color: var(--teal) !important; }

/* Clear button */
.clear-wrap .stButton button {
    background: transparent !important;
    color: var(--txt2) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.7rem !important;
    font-weight: 400 !important;
}
.clear-wrap .stButton button:hover { border-color: var(--txt2) !important; }

/* ── Input ── */
.stTextInput > div > div > input {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--txt) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    height: 42px !important;
    padding: 0 1rem !important;
    caret-color: var(--amber) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--amber) !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(240,165,0,0.15) !important;
}
.stTextInput > div > div > input::placeholder { color: var(--txt2) !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--bg2) !important;
    border-color: var(--border) !important;
    color: var(--txt) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 0.6rem 0.75rem !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--txt2) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1rem !important;
    color: var(--amber) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--amber) !important; }

/* ── Chat messages ── */
.msg-wrap { margin-bottom: 1.75rem; }
.msg-role {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.msg-role.user { color: var(--teal); }
.msg-role.ai   { color: var(--amber); }

.msg-bubble {
    padding: 1rem 1.25rem;
    font-family: 'Fraunces', serif;
    font-size: 0.92rem;
    line-height: 1.75;
    border: 1px solid var(--border);
    border-radius: 0 10px 10px 10px;
    white-space: pre-wrap;
    word-break: break-word;
}
.msg-bubble.user {
    background: #172032;
    border-color: #1f3a5f;
    border-radius: 10px 0 10px 10px;
    max-width: 75%;
    margin-left: auto;
}
.msg-bubble.ai {
    background: var(--bg2);
    max-width: 90%;
}

/* ── Sources ── */
.sources-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--txt2);
    margin: 0.85rem 0 0.5rem;
}
.src-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.4rem;
    border-radius: 0 5px 5px 0;
}
.src-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--amber);
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.src-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: var(--txt2);
    margin-bottom: 0.35rem;
}
.src-preview {
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 0.78rem;
    color: var(--txt2);
    line-height: 1.5;
}
.score-track {
    height: 2px;
    background: var(--border);
    border-radius: 1px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    background: var(--amber);
    border-radius: 1px;
}

/* ── Welcome ── */
.welcome-box {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-top: 3px solid var(--amber);
    padding: 1.75rem 2rem;
    border-radius: 6px;
    margin-bottom: 1.5rem;
}
.welcome-title {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--txt);
    margin-bottom: 0.4rem;
}
.welcome-body {
    font-family: 'Fraunces', serif;
    font-style: italic;
    color: var(--txt2);
    font-size: 0.88rem;
    line-height: 1.65;
}

/* ── Page header ── */
.page-header {
    padding: 1.75rem 0 1.25rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.75rem;
}
.page-title {
    font-family: 'Fraunces', serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--txt);
    margin: 0;
    letter-spacing: -0.02em;
}
.page-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--amber);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000"

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("total_queries", 0),
    ("pending_question", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_health():
    try:
        return requests.get(f"{API_BASE}/health", timeout=5).json()
    except Exception:
        return {"status": "offline"}


def ask_question(question: str, k: int = 3) -> dict:
    r = requests.post(
        f"{API_BASE}/query",
        json={"question": question, "k": k},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def render_sources(sources):
    if not sources:
        return
    st.markdown('<div class="sources-label">— Sources —</div>', unsafe_allow_html=True)
    for s in sources:
        pct = int(s["similarity_score"] * 100)
        st.markdown(f"""
<div class="src-card">
  <div class="src-name">⬡ {s['file_name']}</div>
  <div class="src-meta">chunk {s['chunk_index']} &nbsp;·&nbsp; relevance {pct}%</div>
  <div class="src-preview">{s['content_preview'][:130]}…</div>
  <div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>
</div>""", unsafe_allow_html=True)


def render_message(role, content, sources=None):
    label = "you" if role == "user" else "devdocs·ai"
    st.markdown(f"""
<div class="msg-wrap">
  <div class="msg-role {role}">{label}</div>
  <div class="msg-bubble {role}">{content}</div>
</div>""", unsafe_allow_html=True)
    if sources:
        render_sources(sources)


def run_query(question: str, k: int):
    st.session_state.messages.append({"role": "user", "content": question.strip()})
    with st.spinner(""):
        try:
            resp = ask_question(question.strip(), k=k)
            st.session_state.total_queries += 1
            st.session_state.messages.append({
                "role": "ai",
                "content": resp["answer"],
                "sources": resp["sources"],
            })
        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({
                "role": "ai",
                "content": "Could not reach the API. Is the server running?\n\n"
                           "`uvicorn src.api.main:app --reload`",
                "sources": [],
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "ai",
                "content": f"Error: {e}",
                "sources": [],
            })
    st.rerun()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding-bottom:1rem;border-bottom:1px solid var(--border);margin-bottom:1.25rem">
  <div style="font-family:'JetBrains Mono',monospace;font-size:1rem;
  font-weight:600;color:var(--amber);letter-spacing:0.04em">⬡ DevDocs·AI</div>
  <div style="font-family:'Fraunces',serif;font-style:italic;font-size:0.78rem;
  color:var(--txt2);margin-top:0.2rem">documentation intelligence</div>
</div>
""", unsafe_allow_html=True)

    health = get_health()
    status = health.get("status", "offline")
    dot_color = {"healthy": "#3fb950", "degraded": "#f0a500"}.get(status, "#f85149")

    st.markdown(f"""
<div style="display:flex;align-items:center;gap:7px;margin-bottom:1.25rem">
  <div style="width:8px;height:8px;border-radius:50%;
  background:{dot_color};flex-shrink:0"></div>
  <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
  color:var(--txt2);letter-spacing:0.08em">{status.upper()}</span>
</div>
""", unsafe_allow_html=True)

    if status == "healthy":
        c1, c2 = st.columns(2)
        with c1:
            st.metric("model", health.get("model", "—"))
        with c2:
            st.metric("queries", st.session_state.total_queries)

    st.divider()

    st.markdown("""
<p style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
letter-spacing:0.12em;text-transform:uppercase;color:var(--txt2);
margin:0 0 0.5rem">Retrieval chunks</p>""", unsafe_allow_html=True)

    k_value = st.selectbox(
        "k", [1, 2, 3, 4, 5], index=2,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<div class="clear-wrap">', unsafe_allow_html=True)
    if st.button("clear conversation", use_container_width=True, key="clear"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
<div style="margin-top:2.5rem;text-align:center">
  <p style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
  color:var(--txt2);line-height:1.9">
    Mistral · ChromaDB<br>LangChain · Ollama
  </p>
</div>""", unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1 class="page-title">Developer Documentation Assistant</h1>
  <p class="page-sub">RAG · Local LLM · Source Citations</p>
</div>
""", unsafe_allow_html=True)

# Welcome + example chips
if not st.session_state.messages:
    st.markdown("""
<div class="welcome-box">
  <div class="welcome-title">Ask anything about your documentation.</div>
  <div class="welcome-body">
    Retrieves relevant passages from your docs and generates grounded
    answers with source citations. Every answer traces back to a
    specific document and chunk.
  </div>
</div>
<p style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
letter-spacing:0.12em;text-transform:uppercase;color:var(--txt2);
margin-bottom:0.75rem">Try asking →</p>
""", unsafe_allow_html=True)

    examples = [
        "How do I authenticate with the API?",
        "What error code means invalid token?",
        "How do I create a new user?",
        "What are the rate limits?",
    ]
    c1, c2 = st.columns(2)
    for i, ex in enumerate(examples):
        with (c1 if i % 2 == 0 else c2):
            st.markdown('<div class="example-wrap">', unsafe_allow_html=True)
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_question = ex
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("sources"))

# ── Input bar (always at bottom) ───────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        question = st.text_input(
            "q", placeholder="Ask a question about the documentation...",
            label_visibility="collapsed",
        )
    with cols[1]:
        submitted = st.form_submit_button("Ask →", use_container_width=True)

# ── Dispatch ───────────────────────────────────────────────────────────────────
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    run_query(q, k_value)
elif submitted and question.strip():
    if status != "healthy":
        st.error("API not reachable. Start: `.venv/bin/uvicorn src.api.main:app --reload`")
    else:
        run_query(question, k_value)