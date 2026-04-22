import json
import os

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