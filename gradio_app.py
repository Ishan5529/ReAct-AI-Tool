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
    system_prompt=SystemMessage(f"""You are a reasoning assistant. You must decide whether the question can be answered confidently using general knowledge, or whether external information is required.
        
        Rules:
        1) Answer directly when the question is simple, factual, or commonly known.
        2) Respond to the greetings, if present in the query.
        3) Use real_time_tool to determine the current time before answering any question that depends on, references, or compares against the present time.
        4) When considering previous messages in the conversation, use them only if their timestamps indicate they are still temporally relevant to the current query. If the information is time-sensitive and may be outdated, do not rely on it and instead re-evaluate or fetch fresh information using the appropriate tool.
        5) Use weather_search_tool, only for finding current meteorological data of a city and return summarised output.
        6) Use the web_search_tool only when the answer depends on recent, dynamic, or unverifiable information.
        7) If a meteorological question cannot be fully or confidently answered using only the output of weather_search_tool, then perform a follow-up call to web_search_tool to obtain the missing or clarifying information.
        8) When using a tool, call exactly one tool at a time and provide only the required arguments defined by the tool schema.
        9) After a tool call, read the tool's output and produce a detailed and human-readable final answer.
        10) Do not mention tools, function calls, or internal reasoning in the final answer.
        """)
)

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
        return chat_history, "{}", ""

    query_timestamp = timestamp()
    # Convert chat history to LangChain messages
    messages = []
    for msg in chat_history[-10:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(msg["content"]))

    messages.append(HumanMessage(user_query))

    tool_calls_collected = []
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
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls_collected.extend(msg.tool_calls)
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            final_answer = msg.content

    # Update UI chat history
    chat_history.append({
        "role": "user",
        "content": f"{user_query} <small><small>[{query_timestamp}]</small></small>"
    })
    chat_history.append({
        "role": "assistant",
        "content": f"{final_answer} <small><small>[{answer_timestamp}]</small></small>"
    })

    return chat_history, json.dumps(tool_calls_collected, indent=2), ""


def reset_chat():
    return [], "{}", ""

# -----------------------
# Gradio UI
# -----------------------

with gr.Blocks(title="ReAct AI Chatbot") as demo:
    gr.Markdown("## ReAct AI Chatbot\nMulti-Turn • Tool-Aware")

    chat_state = gr.State([])

    chatbot = gr.Chatbot(
        label="Conversation",
        height=420,
        # type="messages"
    )

    user_input = gr.Textbox(
        label="Your Message",
        placeholder="Ask something...",
        lines=2
    )

    tool_calls_box = gr.Code(
        label="AI Tool Calls (Structured JSON)",
        language="json"
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        reset_btn = gr.Button("Reset Chat")

    send_btn.click(
        fn=chat,
        inputs=[user_input, chat_state],
        outputs=[chatbot, tool_calls_box, user_input]
    )

    reset_btn.click(
        fn=reset_chat,
        outputs=[chatbot, tool_calls_box, user_input]
    )

# -----------------------
# Run
# -----------------------

if __name__ == "__main__":
    demo.launch()
