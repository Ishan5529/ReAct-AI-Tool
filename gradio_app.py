import json
import gradio as gr
from datetime import datetime
import groq
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
from src.model import get_model
from src.tools import get_tools

# -----------------------
# Agent setup
# -----------------------

llm = get_model()
tools = get_tools()

agent = create_agent(
    llm,
    tools=tools,
    system_prompt=SystemMessage(f"""You are a decision-oriented reasoning assistant. Your task is to determine whether a user's query can be answered confidently using general knowledge, or whether clarification and/or external tools are required before proceeding.

GENERAL PRINCIPLES
- Be precise, conservative, and explicit.
- Never assume missing parameters, defaults, user preferences, or interpretations.
- Prefer clarification over incorrect or speculative answers.
- Distinguish clearly between static historical facts and potentially changing records.
- Current date and time is not 2023 or 2024, it is different. Use real_time_tool to find that before answering queries.

DECISION RULES

1) CLARIFICATION FIRST (CRITICAL)
If the user's query lacks sufficient, precise, or unambiguous information (e.g., missing parameters, unclear references, multiple valid interpretations), STOP and ask a concise clarification question before invoking any tool.
- If ambiguity involves a finite or enumerable set (e.g., cities, countries, brands, time ranges, categories), explicitly list the options for the user to choose from.
- Do not proceed until clarification is provided.

2) STATIC vs. DYNAMIC FACTS (MANDATORY CHECK)
Before using any real-time or web tool, classify the query as one of the following:

A) STATIC HISTORICAL FACT
- All-time records
- Retired individuals' achievements
- Completed events
- Well-established historical data that does not change
→ Answer directly without using real_time_tool or web_search_tool.

B) POTENTIALLY DYNAMIC RECORD
- Records held by **currently active individuals**
- Cumulative statistics that may change over time
- Rankings, leaders, or totals without a fixed cutoff date
→ You MUST verify whether the record could have changed since your knowledge cutoff.
→ If verification is required, use real_time_tool (to establish “as of” date) and then web_search_tool if needed.

Never assume a record is static solely because it is phrased as “all-time.”
                                
3) TECHNICAL QUERIES
For technical terms or domains:
- Break the topic into relevant subcategories.
- Enumerate possible options or dimensions (e.g., workload type, platform, constraints).
- Explicitly confirm required parameters.
- Never assume defaults.

4) DIRECT ANSWERS
Answer immediately when the question is:
- Simple
- Factual
- Commonly known
- Unambiguous
- Constant
and does not depend on real-time or dynamic information.

5) GREETINGS
If the query includes a greeting, acknowledge it briefly before addressing the question.

6) TIME SENSITIVITY
- Use real_time_tool to verify the time and date before any query dependent upon time.
- You are trained on data upto 2023-24. Currently, we are much ahead in time.
- If a question depends on or references the present time (e.g., “current,” “latest,” “now,” “today,” “previous”), ALWAYS determine the current time using `real_time_tool` before answering.

7) CONTEXT AWARENESS
When considering previous conversation messages:
- Use them only if their timestamps indicate they are still temporally relevant.
- If the information may be outdated or time-sensitive, do not rely on it; re-evaluate or fetch fresh data using the appropriate tool.

8) WEATHER QUERIES
Use `weather_search_tool` ONLY for current meteorological data.
- Supported locations: cities and countries (ISO 3166-1 alpha-2 codes).
- If a city or country is ambiguous, or multiple pairs exist, ask for clarification before calling the tool.
- Resolve city, country ambiguity by listing explicit choices.
- Don't make assumptions.
- Summarize results concisely.
- If the weather tool output is insufficient or incomplete, perform a follow-up `web_search_tool` call.

9) WEB SEARCH USAGE
Use `web_search_tool` ONLY when the answer depends on recent, dynamic, or otherwise unverifiable information.
Before using it:
- Confirm all missing parameters.
- IMPORTANT: **Use real_time_tool to verify the time and date before any query.**
- Resolve ambiguity by listing explicit choices.
- For technical or product-related queries, confirm use case, constraints, and evaluation criteria.
- Never infer preferences (e.g., gaming vs. AI workloads).

10) TOOL DISCIPLINE
- Make another call, if response is still ambiguous.
- Provide all and only the arguments required by the tool schema.
- Read and fully interpret the tool output before responding.

11) FINAL RESPONSE
- Produce a clear, detailed, and human-readable answer.
- Do NOT mention tools, tool calls, function names, or internal system behavior.
- Do NOT expose internal chain-of-thought.
- If justification is needed, provide a concise, structured explanation in markdown without revealing internal reasoning steps.
""")
)
# - Call exactly one tool at a time.

# -----------------------
# Helper Function
# -----------------------

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

# -----------------------
# Chat handler (NO STREAMING)
# -----------------------

def chat(user_query, chat_history):
    if not user_query.strip():
        return chat_history, "{}", "", ""

    query_timestamp = timestamp()
    # Convert chat history to LangChain messages
    messages = []
    for msg in chat_history[-10:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(msg["content"][:500]))

    messages.append(HumanMessage(user_query))

    tool_calls_collected = []
    reasonings_collected = ""
    final_answer = ""

    ## Handle Rate Limit Error
    try:
        response = agent.invoke({"messages": messages})
    except groq.APIStatusError as e:
        body = getattr(e, "body", None)
        error = body["error"]
        if error and error.get("code") == "rate_limit_exceeded":
            response = {
                "messages": [AIMessage(content = "Rate limit exceeded. Please reduce the message size or try again later.")]
            }
        else:
            raise e
        
    answer_timestamp = timestamp()

    for msg in response["messages"]:
        if isinstance(msg, AIMessage) and msg.additional_kwargs.get('reasoning_content'):
            reasonings_collected += ("\n\n" + msg.additional_kwargs.get('reasoning_content'))
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls_collected.extend(msg.tool_calls)
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            final_answer = msg.content

    if final_answer == "":
        final_answer = "Rate limit exceeded. Please reduce the message size or try again later."

    # Update UI chat history
    chat_history.append({
        "role": "user",
        "content": f"{user_query} <small><small>[{query_timestamp}]</small></small>"
    })
    chat_history.append({
        "role": "assistant",
        "content": f"{final_answer} <small><small>[{answer_timestamp}]</small></small>"
    })

    return chat_history, json.dumps(tool_calls_collected, indent=2), reasonings_collected, ""


def reset_chat():
    return [], "{}", "", "", []

# -----------------------
# Gradio UI
# -----------------------

with gr.Blocks(title="ReAct AI Chatbot") as demo:
    gr.Markdown("## ReAct AI Chatbot\nMulti-Turn • Tool-Aware")

    chat_state = gr.State([])

    chatbot = gr.Chatbot(
        label="Conversation",
        height=420,
    )

    user_input = gr.Textbox(
        label="Your Message",
        placeholder="Ask something...",
        lines=2
    )

    with gr.Row():
        tool_calls_box = gr.Code(
            label="AI Tool Calls (Structured JSON)",
            language="json"
        )
        reasoning_box = gr.Code(
            label="Model Reasoning",
            language="markdown"
        )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        reset_btn = gr.Button("Reset Chat")

    send_btn.click(
        fn=chat,
        inputs=[user_input, chat_state],
        outputs=[chatbot, tool_calls_box, reasoning_box, user_input]
    )

    reset_btn.click(
        fn=reset_chat,
        outputs=[chatbot, tool_calls_box, reasoning_box, user_input, chat_state]
    )

# -----------------------
# Run
# -----------------------

if __name__ == "__main__":
    demo.launch()
