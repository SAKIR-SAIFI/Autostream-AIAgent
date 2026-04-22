import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage
from state import AgentState, LeadCollectionStage
from graph import build_agent

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoStream AI",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e6f0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 760px; }

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ff6b6b 0%, #ff9f43 50%, #ffd32a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.app-header p {
    color: #6b6b80;
    font-size: 0.9rem;
    margin: 0.4rem 0 0;
    font-weight: 300;
}

/* Chat container */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

/* Message bubbles */
.msg-row {
    display: flex;
    align-items: flex-end;
    gap: 0.6rem;
}
.msg-row.user { flex-direction: row-reverse; }

.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.avatar.aria {
    background: linear-gradient(135deg, #ff6b6b, #ff9f43);
}
.avatar.user-av {
    background: #1e1e2e;
    border: 1px solid #2e2e3e;
}

.bubble {
    max-width: 78%;
    padding: 0.75rem 1rem;
    border-radius: 16px;
    font-size: 0.92rem;
    line-height: 1.55;
    word-wrap: break-word;
}
.bubble.aria {
    background: #14141f;
    border: 1px solid #1e1e30;
    border-bottom-left-radius: 4px;
    color: #dcdce8;
}
.bubble.user {
    background: linear-gradient(135deg, #ff6b6b22, #ff9f4322);
    border: 1px solid #ff6b6b33;
    border-bottom-right-radius: 4px;
    color: #e8e6f0;
}

/* Status badge */
.status-bar {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    padding: 0.75rem 1rem;
    background: #0f0f1a;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 0.78rem;
}
.badge {
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-weight: 500;
}
.badge.green  { background: #1a3a1a; color: #4ade80; border: 1px solid #2a5a2a; }
.badge.orange { background: #3a2a0a; color: #fb923c; border: 1px solid #5a3a0a; }
.badge.gray   { background: #1a1a2a; color: #9090a0; border: 1px solid #2a2a3a; }

/* Input area */
.stTextInput > div > div > input {
    background: #0f0f1a !important;
    border: 1px solid #2e2e3e !important;
    border-radius: 10px !important;
    color: #e8e6f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff6b6b !important;
    box-shadow: 0 0 0 2px #ff6b6b22 !important;
}

/* Send button */
.stButton > button {
    background: linear-gradient(135deg, #ff6b6b, #ff9f43) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.65rem 1.4rem !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Reset button */
.reset-btn > button {
    background: transparent !important;
    color: #6b6b80 !important;
    border: 1px solid #2e2e3e !important;
    font-size: 0.8rem !important;
}
.reset-btn > button:hover { color: #ff6b6b !important; border-color: #ff6b6b44 !important; }

/* Typing indicator */
.typing { display: flex; gap: 4px; padding: 0.6rem 0.8rem; }
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #ff6b6b;
    animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%            { transform: translateY(-6px); opacity: 1; }
}

/* Scrollable chat area */
.chat-scroll {
    max-height: 460px;
    overflow-y: auto;
    padding-right: 4px;
    scroll-behavior: smooth;
}
.chat-scroll::-webkit-scrollbar { width: 4px; }
.chat-scroll::-webkit-scrollbar-thumb { background: #2e2e3e; border-radius: 4px; }

/* Welcome card */
.welcome-card {
    text-align: center;
    padding: 3rem 1.5rem;
    color: #4a4a60;
}
.welcome-card .icon { font-size: 3rem; margin-bottom: 1rem; }
.welcome-card p { font-size: 0.9rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
def _initial_agent_state() -> AgentState:
    return {
        "messages": [],
        "intent": None,
        "lead_stage": LeadCollectionStage.NOT_STARTED,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "rag_context": None,
    }

if "agent_state" not in st.session_state:
    st.session_state.agent_state = _initial_agent_state()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"|"aria", "text": str}
if "agent" not in st.session_state:
    with st.spinner("Loading agent…"):
        st.session_state.agent = build_agent()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🎬 AutoStream AI</h1>
    <p>Your intelligent video editing assistant · Powered by LangGraph</p>
</div>
""", unsafe_allow_html=True)

# ── Lead status bar ───────────────────────────────────────────────────────────
s = st.session_state.agent_state
stage = s.get("lead_stage", LeadCollectionStage.NOT_STARTED)

def _badge(label, cls):
    return f'<span class="badge {cls}">{label}</span>'

badges = []
if s.get("lead_name"):
    badges.append(_badge(f"👤 {s['lead_name']}", "green"))
if s.get("lead_email"):
    badges.append(_badge(f"✉ {s['lead_email']}", "green"))
if s.get("lead_platform"):
    badges.append(_badge(f"📱 {s['lead_platform']}", "green"))
if s.get("lead_captured"):
    badges.append(_badge("✅ Lead Captured", "green"))
elif stage != LeadCollectionStage.NOT_STARTED and not s.get("lead_captured"):
    stage_labels = {
        LeadCollectionStage.COLLECTING_NAME: "Collecting name…",
        LeadCollectionStage.COLLECTING_EMAIL: "Collecting email…",
        LeadCollectionStage.COLLECTING_PLATFORM: "Collecting platform…",
        LeadCollectionStage.COMPLETE: "Finalizing…",
    }
    badges.append(_badge(stage_labels.get(stage, stage), "orange"))

if badges:
    st.markdown(
        f'<div class="status-bar">{"".join(badges)}</div>',
        unsafe_allow_html=True
    )

# ── Chat display ──────────────────────────────────────────────────────────────
st.markdown('<div class="chat-scroll"><div class="chat-container">', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown("""
    <div class="welcome-card">
        <div class="icon">🎬</div>
        <p>Hi! I'm <strong style="color:#ff9f43">Aria</strong>, AutoStream's AI assistant.<br>
        Ask me about pricing, features, or getting started!</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="bubble user">{msg["text"]}</div>
                <div class="avatar user-av">🧑</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row">
                <div class="avatar aria">🎬</div>
                <div class="bubble aria">{msg["text"]}</div>
            </div>""", unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True)

# Input row - using st.form prevents re-firing on every rerun
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "message",
            placeholder="Ask about pricing, features, or say hi…",
            label_visibility="collapsed",
        )
    with col2:
        send = st.form_submit_button("Send", use_container_width=True)

# Reset button
st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
if st.button("↺ New conversation", use_container_width=False):
    st.session_state.agent_state = _initial_agent_state()
    st.session_state.chat_history = []
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Handle send
if send and user_input.strip():
    text = user_input.strip()

    # Append user message to display history
    st.session_state.chat_history.append({"role": "user", "text": text})

    # Update agent state with new human message
    current = st.session_state.agent_state
    current["messages"] = current["messages"] + [HumanMessage(content=text)]

    # Run graph
    with st.spinner(""):
        new_state = st.session_state.agent.invoke(current)

    st.session_state.agent_state = new_state

    # Extract last AI message
    last_ai = ""
    for msg in reversed(new_state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai = msg.content
            break

    if last_ai:
        st.session_state.chat_history.append({"role": "aria", "text": last_ai})

    st.rerun()