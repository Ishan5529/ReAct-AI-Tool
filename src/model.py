import os
from langchain_groq import ChatGroq

# -----------------------
# LLM
# -----------------------
def get_model():
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        # model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=2000,
        timeout=30,
        max_retries=2,
    )

    return llm