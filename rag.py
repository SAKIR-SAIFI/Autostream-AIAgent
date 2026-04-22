import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Build documents from knowledge base ──────────────────────────────────────

def load_knowledge_base() -> dict:
    """Load the AutoStream knowledge base from JSON."""
    kb_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd(),
        "knowledge_base.json"
    )
    with open(kb_path, "r") as f:
        return json.load(f)


def _build_documents(kb: dict) -> list[Document]:
    """Convert the knowledge base dict into a flat list of LangChain Documents."""
    docs = []

    # Pricing
    for plan_key, plan in kb["pricing"].items():
        text = (
            f"{plan['name']}: {plan['price']}. "
            f"Features: {', '.join(plan['features'])}."
        )
        docs.append(Document(page_content=text, metadata={"section": "pricing", "plan": plan_key}))

    # Policies
    for policy_key, policy_val in kb["policies"].items():
        docs.append(Document(
            page_content=f"{policy_key.replace('_', ' ').title()}: {policy_val}",
            metadata={"section": "policies", "policy": policy_key}
        ))

    # Product info
    product = kb["product_info"]
    docs.append(Document(
        page_content=f"About AutoStream: {product['description']}",
        metadata={"section": "product_info", "type": "description"}
    ))
    docs.append(Document(
        page_content=f"AutoStream use cases: {', '.join(product['use_cases'])}.",
        metadata={"section": "product_info", "type": "use_cases"}
    ))
    docs.append(Document(
        page_content=f"AutoStream key features: {', '.join(product['key_features'])}.",
        metadata={"section": "product_info", "type": "features"}
    ))

    return docs


# ── Build FAISS index at import time ─────────────────────────────────────────

_KB = load_knowledge_base()
_DOCS = _build_documents(_KB)

# Using a lightweight local embedding model — no API key required.
# Swap to langchain_openai.OpenAIEmbeddings() if you prefer.
_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

_VECTOR_STORE = FAISS.from_documents(_DOCS, _EMBEDDINGS)


# ── Public interface (same signatures as before) ──────────────────────────────

def retrieve_knowledge(query: str, k: int = 3) -> str:
    """
    FAISS-powered semantic retrieval: finds the top-k most relevant
    chunks from the knowledge base for the given query.
    Returns a formatted context string.
    """
    results: list[tuple[Document, float]] = _VECTOR_STORE.similarity_search_with_score(query, k=k)

    if not results:
        return ""

    section_order = {"pricing": 0, "policies": 1, "product_info": 2}
    seen_sections: dict[str, list[str]] = {}

    for doc, score in results:
        section = doc.metadata.get("section", "general")
        seen_sections.setdefault(section, []).append(doc.page_content)

    context_parts = []
    for section in sorted(seen_sections, key=lambda s: section_order.get(s, 99)):
        header = {
            "pricing": "PRICING INFO",
            "policies": "POLICY INFO",
            "product_info": "PRODUCT INFO",
        }.get(section, section.upper())
        context_parts.append(f"{header}:\n" + "\n".join(seen_sections[section]))

    return "\n\n".join(context_parts)