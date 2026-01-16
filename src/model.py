import os
from langchain_groq import ChatGroq

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Check your .env file location and contents.")

# -----------------------
# LLM
# -----------------------
def get_model():
    llm = ChatGroq(
        # model="qwen/qwen3-32b",
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1000,
        timeout=30,
        max_retries=2,
    )

    return llm