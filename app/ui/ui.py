import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/query"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.set_page_config(
    page_title="SEBI Command Deck",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --bg: #121212;
    --bg-soft: #1c1c1f;
    --panel: #232327;
    --panel-soft: #2e2e33;
    --line: #47474f;
    --ink: #f7efe0;
    --muted: #d6c7ae;
    --good: #c0222c;
    --warn: #db9f3b;
    --accent: #ff4d4f;
    --accent-2: #ffb347;
    --accent-3: #ffd166;
    --accent-4: #4da8ff;
}

.stApp {
    background:
        radial-gradient(1100px 650px at -10% -25%, rgba(255, 77, 79, 0.24), transparent 62%),
        radial-gradient(1000px 600px at 110% -15%, rgba(255, 179, 71, 0.20), transparent 62%),
        radial-gradient(900px 540px at 45% 120%, rgba(77, 168, 255, 0.18), transparent 62%),
        linear-gradient(180deg, #0f0f0f 0%, var(--bg) 100%);
    color: var(--ink);
}

.block-container {
    max-width: 920px;
    margin: 0 auto;
    padding-top: 0.9rem;
    padding-bottom: 2rem;
}

.hero {
    border-radius: 14px;
    padding: 1rem 1.05rem;
    color: var(--ink);
    border: 1px solid var(--line);
    background: linear-gradient(130deg, #251919 0%, #5a1d2a 45%, #7a3f1d 100%);
    box-shadow: none;
}

.hero h1 {
    margin: 0;
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: 0;
}

.hero p {
    margin: 0.35rem 0 0 0;
    color: var(--muted);
    font-size: 0.95rem;
}

.ribbon {
    margin-top: 0.7rem;
    display: inline-block;
    background: linear-gradient(120deg, rgba(255, 77, 79, 0.38), rgba(255, 179, 71, 0.34));
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--muted);
    padding: 0.22rem 0.6rem;
    font-size: 0.78rem;
}

.stats {
    margin-top: 0.7rem;
    display: grid;
    gap: 0.5rem;
    grid-template-columns: repeat(4, minmax(0, 1fr));
}

.stat {
    background: linear-gradient(160deg, rgba(62, 29, 36, 0.92), rgba(56, 39, 24, 0.92));
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.55rem 0.68rem;
    box-shadow: none;
}

.k {
    margin: 0;
    color: var(--muted);
    font-size: 0.74rem;
}

.v {
    margin: 0.12rem 0 0 0;
    color: var(--ink);
    font-weight: 600;
}

[data-testid="stChatMessage"] {
    border-radius: 12px;
    border: 1px solid var(--line);
    background: linear-gradient(145deg, rgba(42, 29, 30, 0.92), rgba(30, 30, 35, 0.94));
    box-shadow: none;
    color: var(--ink);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {
    color: var(--ink) !important;
}

.evidence {
    margin-bottom: 0.7rem;
    border-radius: 12px;
    border: 1px solid var(--line);
    background: linear-gradient(145deg, rgba(45, 35, 29, 0.92), rgba(34, 34, 38, 0.94));
    padding: 0.72rem 0.78rem;
    color: var(--ink);
}

.eh {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.38rem;
    gap: 0.8rem;
}

.badge {
    border-radius: 999px;
    font-size: 0.74rem;
    padding: 0.14rem 0.52rem;
    color: #fff;
    background: linear-gradient(120deg, var(--accent), #d7263d);
}

.mode {
    border-radius: 999px;
    font-size: 0.72rem;
    padding: 0.16rem 0.52rem;
    color: #fff;
}

.mode-llm {
    background: linear-gradient(120deg, #ff5a5f, #ff7a45);
}

.mode-fallback {
    background: linear-gradient(120deg, #ffb347, #ffd166);
    color: #2d1b05;
}

[data-testid="stChatInput"] {
    background: linear-gradient(145deg, rgba(61, 30, 34, 0.9), rgba(39, 39, 44, 0.95));
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.22rem 0.45rem;
    box-shadow: none;
}

.empty-state {
    margin-top: 1rem;
    border-radius: 12px;
    border: 1px dashed var(--line);
    background: linear-gradient(145deg, rgba(53, 33, 34, 0.9), rgba(37, 37, 41, 0.94));
    color: var(--muted);
    padding: 1rem;
    text-align: center;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171117 0%, #1f1611 100%);
    border-right: 1px solid #413028;
}

[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}

[data-testid="stCodeBlock"] {
    border-radius: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "mode" not in st.session_state:
    st.session_state.mode = "unknown"


@st.cache_data(ttl=10)
def _check_api_health() -> str:
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        return "online" if r.status_code == 200 else "degraded"
    except requests.RequestException:
        return "offline"


health = _check_api_health()

st.markdown(
    """
<div class="hero">
  <h1>SEBI Assistant</h1>
  <p>Ask anything about SEBI regulations and get evidence-backed answers.</p>
  <span class="ribbon">SEBI RAG</span>
</div>
""",
    unsafe_allow_html=True,
)

stats = f"""
<div class="stats">
  <div class="stat"><p class="k">API status</p><p class="v">{health}</p></div>
  <div class="stat"><p class="k">Messages</p><p class="v">{len(st.session_state.messages)}</p></div>
  <div class="stat"><p class="k">Turns</p><p class="v">{len(st.session_state.history)}</p></div>
  <div class="stat"><p class="k">Last mode</p><p class="v">{st.session_state.mode}</p></div>
</div>
"""
st.markdown(stats, unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.markdown("### Session")
    st.caption("Reset or inspect runtime details.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.mode = "unknown"
        st.rerun()
    st.markdown("---")
    st.markdown("### Runtime")
    st.code(API_URL)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if not st.session_state.messages:
    st.markdown(
        "<div class='empty-state'>Ask a SEBI question below to start the conversation.</div>",
        unsafe_allow_html=True,
    )

query = st.chat_input("Ask a SEBI regulation question...")

if query and query.strip():
    query = query.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Processing regulatory context..."):
        payload = {"question": query, "history": st.session_state.history}
        try:
            response = requests.post(API_URL, json=payload, timeout=90)
        except requests.RequestException as exc:
            response = None
            st.error(f"Could not reach API: {exc}")

    if response is not None and response.status_code == 200:
        data = response.json()
        answer = data.get("answer", "")
        evidence = data.get("evidence", [])
        mode = data.get("mode", "unknown")

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.history.append({"user": query, "assistant": answer})
        st.session_state.mode = mode

        with st.chat_message("assistant"):
            mode_class = "mode-fallback" if mode == "fallback" else "mode-llm"
            st.markdown(
                f"<span class='mode {mode_class}'>mode: {mode}</span>",
                unsafe_allow_html=True,
            )
            st.write(answer)

        if evidence:
            with st.expander("Evidence", expanded=True):
                for item in evidence:
                    source = item.get("source_file", "unknown")
                    chunk_index = item.get("chunk_index", "n/a")
                    text = item.get("text", "")
                    score = item.get("score")
                    badge = ""
                    if score is not None:
                        badge = f"<span class='badge'>score {score:.4f}</span>"

                    st.markdown(
                        f"""
<div class="evidence">
  <div class="eh">
    <strong>{source} - chunk {chunk_index}</strong>
    {badge}
  </div>
  <div>{text}</div>
</div>
""",
                        unsafe_allow_html=True,
                    )
    elif response is not None:
        st.error(f"API error: {response.status_code} - {response.text}")
