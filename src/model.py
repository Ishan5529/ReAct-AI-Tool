import os
from langchain_groq import ChatGroq

# -----------------------
# LLM
# -----------------------
def get_model(tokens = 2000):
    llm = ChatGroq(
        # model="qwen/qwen3-32b",
        # model="llama-3.1-8b-instant",
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=tokens,
        timeout=30,
        max_retries=2,
    )

    return llm