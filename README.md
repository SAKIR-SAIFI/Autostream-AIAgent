# 🎬 AutoStream AI Agent

A production-grade Conversational AI Agent for AutoStream — a SaaS video editing platform — built with **LangGraph**, featuring intent detection, RAG-powered knowledge retrieval, and automated lead capture.

Built as part of the **ServiceHive / Inflx Machine Learning Intern Assignment**.

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

Create a `.env` file in the project root:

```env
# Pick ONE of the following:
GOOGLE_API_KEY=your_gemini_api_key_here       # Recommended (free tier)
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Get a **free** Gemini API key at: https://aistudio.google.com/app/apikey

> The agent auto-detects which key is present and picks the right LLM.

### 5. Run the agent

```bash
python agent.py
```

---

## 💬 Sample Conversation

```
You: Hi there!
Aria: Hey! Welcome to AutoStream 👋 I'm Aria, your AI assistant...

You: What are your pricing plans?
Aria: We have two plans: Basic at $29/month (10 videos, 720p)...

You: That sounds great, I want to try the Pro plan for my YouTube channel!
Aria: Awesome! I'd love to get you started. What's your name?

You: Alex Johnson
Aria: Nice to meet you, Alex! What's your email address?

You: alex@example.com
Aria: Got it! Which platform do you primarily create content for?

You: YouTube
✅ Lead captured successfully: Alex Johnson, alex@example.com, YouTube
Aria: You're all set, Alex! Our team will reach out shortly...
```

---

## 🏗️ Architecture Explanation (~200 words)

The agent is built on **LangGraph**, a framework for constructing stateful, multi-step AI workflows as directed graphs. LangGraph was chosen over AutoGen because it offers fine-grained control over conversation flow, explicit state management, and deterministic routing between nodes — essential for a production lead-capture system where tool calls must be triggered only at the right moment.

**How state is managed:** A single `AgentState` TypedDict holds the entire session: the full conversation history (as LangChain `BaseMessage` objects), detected intent, lead collection stage, and collected lead fields (name, email, platform). LangGraph's `add_messages` annotation ensures new messages are appended (not replaced) across turns, giving the agent memory across 5–6+ conversation turns without any external store.

**Graph flow:** Each user message enters at `detect_intent` → conditional routing sends it to `rag_retrieval` (product questions), `collect_lead` (high-intent or mid-collection), or directly to `generate_response` (greetings). The lead collection node uses a finite state machine (`NOT_STARTED → COLLECTING_NAME → COLLECTING_EMAIL → COLLECTING_PLATFORM → COMPLETE`) to extract values one at a time. Only when all three are collected does the graph route to `execute_lead_capture`, which fires `mock_lead_capture()`. This prevents premature tool calls by design.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, use the **WhatsApp Business Cloud API** (Meta):

### Architecture

```
User (WhatsApp) 
    │
    ▼
Meta WhatsApp Webhook (POST /webhook)
    │
    ▼
FastAPI / Flask Server
    │  - Verify webhook token on GET
    │  - Parse inbound message on POST
    │  - Load session state from Redis/DB
    │  - Invoke LangGraph agent
    │  - Save updated state back to Redis/DB
    ▼
LangGraph Agent (agent.py)
    │
    ▼
Meta Send Message API (reply to user)
```

### Key Implementation Steps

1. **Register Webhook** on Meta Developer Portal — point it to your server's `/webhook` endpoint.

2. **Verify Webhook** (GET request): Check `hub.verify_token` matches your secret.

3. **Handle Inbound Messages** (POST request):
   ```python
   @app.post("/webhook")
   async def handle_webhook(payload: dict):
       phone = payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
       text  = payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
       
       # Load state from Redis (keyed by phone number)
       state = redis.get(f"session:{phone}") or initial_state()
       
       # Run LangGraph agent
       state["messages"].append(HumanMessage(content=text))
       new_state = agent.invoke(state)
       
       # Save state back
       redis.set(f"session:{phone}", new_state, ex=3600)
       
       # Send reply via Meta API
       send_whatsapp_message(phone, get_last_ai_message(new_state))
   ```

4. **Session Persistence**: Use **Redis** or a database to store `AgentState` per phone number — this replaces the in-memory dict from the CLI version and enables multi-user concurrency.

5. **Deploy**: Host on AWS/GCP/Render with HTTPS (required by Meta for webhooks).

---

## 📁 Project Structure

```
autostream-agent/
├── agent.py              # Main LangGraph agent (all nodes + graph)
├── knowledge_base.json   # RAG knowledge base (pricing, policies, features)
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
└── README.md             # This file
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Gemini 1.5 Flash API key (recommended, free) |
| `OPENAI_API_KEY` | GPT-4o-mini API key (alternative) |
| `ANTHROPIC_API_KEY` | Claude Haiku API key (alternative) |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph 0.2+ |
| LLM | Gemini 1.5 Flash / GPT-4o-mini / Claude Haiku |
| RAG | Custom JSON knowledge base + keyword retrieval |
| State Management | LangGraph `StateGraph` + `TypedDict` |
| Language | Python 3.9+ |

---

## 📊 Evaluation Checklist

- ✅ Intent detection (greeting / product inquiry / high intent)
- ✅ RAG-powered knowledge retrieval from local JSON
- ✅ Multi-turn memory across 5–6 conversation turns
- ✅ Progressive lead collection (name → email → platform)
- ✅ `mock_lead_capture()` only fires after all 3 values collected
- ✅ Clean LangGraph state machine architecture
- ✅ WhatsApp webhook deployment plan documented
