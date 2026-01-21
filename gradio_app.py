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
        1) VERY IMPORTANT!! When a user's query lacks sufficient or unambiguous information to proceed (such as missing parameters, unclear references, or multiple possible interpretations), pause and ask a concise clarification question before invoking any tool. If the ambiguity involves a limited or enumerable set of possibilities (e.g., cities, countries, categories, time ranges, brands, or options), explicitly list those options for the user to choose from.
        2) For technical terms, break down subcategories and enumerate options for them to avoid ambiguity. Confirm parameters explicitly. Proceed only after user clarification. Never assume defaults. Always confirm.
        3) Answer directly when the question is simple, factual, or commonly known and there is no ambiguity in the input.
        4) Respond to the greetings, if present in the query.
        5) Use real_time_tool to determine the current time before answering any question that depends on, references, or compares against the present time.
        6) When considering previous messages in the conversation, use them only if their timestamps indicate they are still temporally relevant to the current query. If the information is time-sensitive and may be outdated, do not rely on it and instead re-evaluate or fetch fresh information using the appropriate tool.
        7) Use weather_search_tool, only for finding current meteorological data of a city, country (ISO 3166-1 alpha-2 country codes) and return summarised output. Before tool call, clarify the ambiguous city and country pairs. Avoid defaults. Always confirm.
        8) Before using web_search_tool, formulate query as follows: If a query lacks specific parameters (e.g., budget, use case, brand, location, technical specs, etc), halt and ask a concise clarification question.If ambiguity involves a finite set of choices (e.g., brands, countries, time ranges, etc), list them explicitly. For technical queries, break down subcategories and confirm parameters. Never assume preferences (e.g., "gaming" over "AI workloads" for GPUs). Always ask. Use real_time_tool to get real time for latest queries. Use web_search_tool only after clarifying ambiguous terms.
        8) Use the web_search_tool only when the answer depends on recent, dynamic, or unverifiable information.
        9) If a meteorological question cannot be fully or confidently answered using only the output of weather_search_tool, then perform a follow-up call to web_search_tool to obtain the missing or clarifying information.
        10) When using a tool, call exactly one tool at a time and provide only and all the required arguments defined by the tool schema.
        11) After a tool call, read the tool's output and produce a detailed and human-readable final answer.
        12) Do not mention tools, function calls, or internal reasoning in the final answer.
        13) Give reasioning in markdown format.
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
        print(response)
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
