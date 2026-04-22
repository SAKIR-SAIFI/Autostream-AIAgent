
from langchain_core.messages import HumanMessage, AIMessage
from graph import build_agent, AgentState
from state import LeadCollectionStage

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
