
from langgraph.graph import StateGraph, END
from state import AgentState
from router import route_after_intent, route_after_lead_collection
from nodes import detect_intent, rag_retrieval, collect_lead_info, execute_lead_capture, generate_response 

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