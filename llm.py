
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    """
    Returns the LLM. Tries Gemini first (free tier available),
    falls back to a helpful error message.
    Set GOOGLE_API_KEY in your environment, OR swap for OpenAI/Anthropic below.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(model="openai/gpt-oss-120b", temperature=0.3, api_key=groq_key)

    # Fallback: Gemini (free tier available)
    google_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if google_api_key:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
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
        "  GROQ_API_KEY (gpt-oss-120b)"
    )