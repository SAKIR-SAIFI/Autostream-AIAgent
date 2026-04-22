from state import AgentState, IntentType, LeadCollectionStage


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