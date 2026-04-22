import re

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from llm import get_llm
from rag import retrieve_knowledge
from tools import mock_lead_capture
from state import AgentState, IntentType, LeadCollectionStage


LLM = get_llm()

# --- Node 1: Intent Detection ---

INTENT_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, a video editing SaaS.

Classify the user's latest message into EXACTLY ONE of:
- greeting        → simple hello, hi, hey, how are you, etc.
- product_inquiry → asking about features, pricing, plans, policies, comparisons
- high_intent     → clearly wants to sign up, try, buy, subscribe, or start

Reply with ONLY the label (one word, lowercase). No explanation."""

def detect_intent(state: AgentState) -> AgentState:
    """Classify the user's intent from their latest message."""
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    if not last_human_msg:
        return {**state, "intent": IntentType.UNKNOWN}

    response = LLM.invoke([
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=last_human_msg)
    ])

    raw = response.content.strip().lower()

    if "high" in raw or "intent" in raw:
        intent = IntentType.HIGH_INTENT
    elif "product" in raw or "inquiry" in raw:
        intent = IntentType.PRODUCT_INQUIRY
    elif "greeting" in raw or "greet" in raw:
        intent = IntentType.GREETING
    else:
        intent = IntentType.UNKNOWN

    print(f"[Intent Detected]: {intent}")
    return {**state, "intent": intent}


# --- Node 2: RAG Retrieval ---

def rag_retrieval(state: AgentState) -> AgentState:
    """Retrieve relevant knowledge base context for the current query."""
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    context = retrieve_knowledge(last_human_msg)
    print(f"[RAG Context Retrieved]: {len(context)} chars")
    return {**state, "rag_context": context}


# --- Node 3: Lead Collection ---

def collect_lead_info(state: AgentState) -> AgentState:
    """
    Progressively collect lead info: name → email → platform.
    Extracts values from the user's last message based on current stage.
    """
    last_human_msg = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break

    stage = state.get("lead_stage", LeadCollectionStage.NOT_STARTED)
    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")

    # Extract data based on stage
    if stage == LeadCollectionStage.COLLECTING_NAME:
        # Use the user's message as their name (clean it up)
        extracted = last_human_msg.strip().strip("\"'")
        if len(extracted) > 1:
            name = extracted
            stage = LeadCollectionStage.COLLECTING_EMAIL

    elif stage == LeadCollectionStage.COLLECTING_EMAIL:
        # Extract email via regex
        email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", last_human_msg)
        if email_match:
            email = email_match.group(0)
            stage = LeadCollectionStage.COLLECTING_PLATFORM

    elif stage == LeadCollectionStage.COLLECTING_PLATFORM:
        # Accept any platform mention
        known_platforms = ["youtube", "instagram", "tiktok", "twitter", "x", "facebook", "linkedin", "twitch", "snapchat"]
        msg_lower = last_human_msg.lower()
        for p in known_platforms:
            if p in msg_lower:
                platform = p.capitalize()
                break
        if not platform:
            platform = last_human_msg.strip()  # Accept free-form input
        stage = LeadCollectionStage.COMPLETE

    elif stage == LeadCollectionStage.NOT_STARTED:
        # High intent detected — begin collection
        stage = LeadCollectionStage.COLLECTING_NAME

    return {
        **state,
        "lead_stage": stage,
        "lead_name": name,
        "lead_email": email,
        "lead_platform": platform,
    }


# --- Node 4: Tool Execution (Lead Capture) ---

def execute_lead_capture(state: AgentState) -> AgentState:
    """Fire the mock_lead_capture tool once all 3 values are collected."""
    result = mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"]
    )
    return {**state, "lead_captured": True}


# --- Node 5: Response Generation ---

AGENT_SYSTEM_PROMPT = """You are Aria, the friendly and knowledgeable sales assistant for AutoStream — an AI-powered video editing SaaS for content creators.

Your personality:
- Warm, helpful, and professional
- Concise but thorough (2-4 sentences per response)
- Enthusiastic about AutoStream's capabilities
- Never pushy — let the user lead

Your capabilities:
- Answer questions about AutoStream using the knowledge base context provided
- Identify when users are ready to sign up and guide them through lead capture
- Keep conversation history in mind for continuity

Always use the KNOWLEDGE BASE CONTEXT if it's relevant to the user's question.
Never fabricate pricing, features, or policies — only use what's in the context.

Current lead collection status is provided to you. Guide accordingly:
- NOT_STARTED or greeting: just chat normally
- COLLECTING_NAME: politely ask for their name
- COLLECTING_EMAIL: ask for their email address  
- COLLECTING_PLATFORM: ask which platform they create content for
- COMPLETE: thank them and confirm everything is set
"""

def generate_response(state: AgentState) -> AgentState:
    """Generate the agent's reply using full conversation history + context."""
    intent = state.get("intent", IntentType.UNKNOWN)
    stage = state.get("lead_stage", LeadCollectionStage.NOT_STARTED)
    lead_captured = state.get("lead_captured", False)
    rag_context = state.get("rag_context", "")

    # Build the contextual system message for this turn
    context_block = ""
    if rag_context:
        context_block += f"\n\nKNOWLEDGE BASE CONTEXT:\n{rag_context}"

    context_block += f"\n\nLEAD COLLECTION STATUS: {stage}"
    if state.get("lead_name"):
        context_block += f"\nCollected name: {state['lead_name']}"
    if state.get("lead_email"):
        context_block += f"\nCollected email: {state['lead_email']}"
    if state.get("lead_platform"):
        context_block += f"\nCollected platform: {state['lead_platform']}"

    if lead_captured:
        context_block += "\n\nACTION: Lead has just been successfully captured. Confirm and wrap up warmly."

    system_msg = SystemMessage(content=AGENT_SYSTEM_PROMPT + context_block)

    # Build messages list for LLM (system + full history)
    llm_messages = [system_msg] + list(state["messages"])

    response = LLM.invoke(llm_messages)

    return {**state, "messages": [AIMessage(content=response.content)]}