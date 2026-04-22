"""
AutoStream Conversational AI Agent
Built with LangGraph for ServiceHive / Inflx Assignment
"""

import json
import os
import re
from typing import Annotated, TypedDict, Optional
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE (RAG)
# ─────────────────────────────────────────────

def load_knowledge_base() -> dict:
    """Load the AutoStream knowledge base from JSON."""
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    with open(kb_path, "r") as f:
        return json.load(f)

KNOWLEDGE_BASE = load_knowledge_base()

def retrieve_knowledge(query: str) -> str:
    """
    Simple RAG retrieval: search the knowledge base for relevant info
    based on keywords in the user query.
    Returns a formatted context string.
    """
    query_lower = query.lower()
    context_parts = []

    # Pricing keywords
    if any(k in query_lower for k in ["price", "pricing", "cost", "plan", "basic", "pro", "pay", "subscription", "how much"]):
        pricing = KNOWLEDGE_BASE["pricing"]
        basic = pricing["basic_plan"]
        pro = pricing["pro_plan"]
        context_parts.append(
            f"PRICING INFO:\n"
            f"Basic Plan: {basic['price']} — {', '.join(basic['features'])}\n"
            f"Pro Plan: {pro['price']} — {', '.join(pro['features'])}"
        )

    # Policy keywords
    if any(k in query_lower for k in ["refund", "cancel", "support", "trial", "policy", "help", "return"]):
        policies = KNOWLEDGE_BASE["policies"]
        context_parts.append(
            f"POLICY INFO:\n"
            f"Refunds: {policies['refund_policy']}\n"
            f"Support: {policies['support']}\n"
            f"Trial: {policies['trial']}\n"
            f"Cancellation: {policies['cancellation']}"
        )

    # Product/feature keywords
    if any(k in query_lower for k in ["feature", "what is", "autostream", "edit", "video", "caption", "4k", "resolution", "template", "platform"]):
        product = KNOWLEDGE_BASE["product_info"]
        context_parts.append(
            f"PRODUCT INFO:\n"
            f"About: {product['description']}\n"
            f"Use cases: {', '.join(product['use_cases'])}\n"
            f"Key features: {', '.join(product['key_features'])}"
        )

    if not context_parts:
        # Default: return all pricing as fallback
        pricing = KNOWLEDGE_BASE["pricing"]
        basic = pricing["basic_plan"]
        pro = pricing["pro_plan"]
        context_parts.append(
            f"GENERAL INFO:\n"
            f"Basic Plan: {basic['price']} — {', '.join(basic['features'])}\n"
            f"Pro Plan: {pro['price']} — {', '.join(pro['features'])}\n"
            f"Trial: {KNOWLEDGE_BASE['policies']['trial']}"
        )

    return "\n\n".join(context_parts)


# ─────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API call to capture a qualified lead.
    In production, this would POST to a CRM like HubSpot or Salesforce.
    """
    print(f"\n{'='*50}")
    print(f"✅ Lead captured successfully: {name}, {email}, {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# 3. AGENT STATE
# ─────────────────────────────────────────────

class IntentType(str, Enum):
    GREETING = "greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    HIGH_INTENT = "high_intent"
    UNKNOWN = "unknown"

class LeadCollectionStage(str, Enum):
    NOT_STARTED = "not_started"
    COLLECTING_NAME = "collecting_name"
    COLLECTING_EMAIL = "collecting_email"
    COLLECTING_PLATFORM = "collecting_platform"
    COMPLETE = "complete"

class AgentState(TypedDict):
    # Full conversation history (LangGraph managed)
    messages: Annotated[list[BaseMessage], add_messages]
    # Detected intent of the latest user message
    intent: Optional[str]
    # Lead collection progress
    lead_stage: str
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    # Whether lead has been captured
    lead_captured: bool
    # RAG context for current turn
    rag_context: Optional[str]


# ─────────────────────────────────────────────
# 4. LLM SETUP
# ─────────────────────────────────────────────

def get_llm():
    """
    Returns the LLM. Tries Gemini first (free tier available),
    falls back to a helpful error message.
    Set GOOGLE_API_KEY in your environment, OR swap for OpenAI/Anthropic below.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

    # Fallback: OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=openai_key)

    # Fallback: Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5", temperature=0.3, api_key=anthropic_key)

    raise EnvironmentError(
        "No LLM API key found. Please set one of:\n"
        "  GOOGLE_API_KEY (Gemini 1.5 Flash)\n"
        "  OPENAI_API_KEY (GPT-4o-mini)\n"
        "  ANTHROPIC_API_KEY (Claude Haiku)"
    )

LLM = get_llm()


# ─────────────────────────────────────────────
# 5. GRAPH NODES
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# 6. ROUTING LOGIC
# ─────────────────────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """Route to RAG, lead collection, or direct response."""
    intent = state.get("intent", IntentType.UNKNOWN)
    stage = state.get("lead_stage", LeadCollectionStage.NOT_STARTED)

    # If we're already in a lead collection flow, continue it
    if stage in [
        LeadCollectionStage.COLLECTING_NAME,
        LeadCollectionStage.COLLECTING_EMAIL,
        LeadCollectionStage.COLLECTING_PLATFORM,
    ]:
        return "collect_lead"

    if intent == IntentType.HIGH_INTENT:
        return "collect_lead"
    elif intent == IntentType.PRODUCT_INQUIRY:
        return "rag_retrieval"
    else:
        return "generate_response"


def route_after_lead_collection(state: AgentState) -> str:
    """After updating lead stage, either capture or respond."""
    stage = state.get("lead_stage", LeadCollectionStage.NOT_STARTED)
    if stage == LeadCollectionStage.COMPLETE:
        return "execute_capture"
    return "generate_response"


# ─────────────────────────────────────────────
# 7. BUILD THE LANGGRAPH
# ─────────────────────────────────────────────

def build_agent():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("detect_intent", detect_intent)
    graph.add_node("rag_retrieval", rag_retrieval)
    graph.add_node("collect_lead", collect_lead_info)
    graph.add_node("execute_capture", execute_lead_capture)
    graph.add_node("generate_response", generate_response)

    # Entry point
    graph.set_entry_point("detect_intent")

    # Conditional routing after intent detection
    graph.add_conditional_edges(
        "detect_intent",
        route_after_intent,
        {
            "collect_lead": "collect_lead",
            "rag_retrieval": "rag_retrieval",
            "generate_response": "generate_response",
        }
    )

    # After RAG → generate response
    graph.add_edge("rag_retrieval", "generate_response")

    # After lead collection → decide: capture or respond
    graph.add_conditional_edges(
        "collect_lead",
        route_after_lead_collection,
        {
            "execute_capture": "execute_capture",
            "generate_response": "generate_response",
        }
    )

    # After lead capture → generate response
    graph.add_edge("execute_capture", "generate_response")

    # End after response
    graph.add_edge("generate_response", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 8. MAIN CHAT LOOP
# ─────────────────────────────────────────────

def run_agent():
    """Interactive CLI chat loop with the AutoStream agent."""
    agent = build_agent()

    print("\n" + "="*60)
    print("  🎬  AutoStream AI Assistant (Powered by LangGraph)")
    print("="*60)
    print("  Type 'quit' or 'exit' to end the conversation.")
    print("="*60 + "\n")

    # Initial state
    state: AgentState = {
        "messages": [],
        "intent": None,
        "lead_stage": LeadCollectionStage.NOT_STARTED,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "rag_context": None,
    }

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nAria: Thanks for chatting! Have a great day. 🎬\n")
            break

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph
        state = agent.invoke(state)

        # Print agent response
        last_ai_msg = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_msg = msg.content
                break

        print(f"\nAria: {last_ai_msg}\n")


if __name__ == "__main__":
    run_agent()
