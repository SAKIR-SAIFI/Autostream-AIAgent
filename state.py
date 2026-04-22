from typing import Annotated, TypedDict, Optional
from enum import Enum

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

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

