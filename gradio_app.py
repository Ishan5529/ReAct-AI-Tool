from dotenv import load_dotenv
load_dotenv()

import json
import gradio as gr
import groq
from src.agent import setup_agent

from langchain.messages import (
    HumanMessage,
    AIMessage
)
from rag_summarize import summarize

# # -----------------------
# # Agent setup
# # -----------------------
agent = setup_agent()

# # -----------------------
# # Chat handler (NO STREAMING)
# # -----------------------

def chat(user_query, chat_history, chat_context):
    if not user_query.strip():
        return chat_history, "{}", "", "", chat_context

    messages = []
    messages.append(HumanMessage(f"PREVIOUS CONTEXT: {chat_context}\n\nCURRENT_QUERY: {user_query}"))

    tool_calls_collected = []
    reasonings_collected = ""
    final_answer = ""

    ## Handle Rate Limit Error
    try:
        response = agent.invoke({"messages": messages})
    except groq.APIStatusError as e:
        body = getattr(e, "body", None)
        error = body["error"]
        print(error)
        if error and error.get("code") == "rate_limit_exceeded":
            response = {
                "messages": [AIMessage(content = "")]
            }
        else:
            raise e

    for msg in response["messages"]:
        if isinstance(msg, AIMessage) and msg.additional_kwargs.get('reasoning_content'):
            reasonings_collected += ("\n\n" + msg.additional_kwargs.get('reasoning_content'))
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls_collected.extend(msg.tool_calls)
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            final_answer = msg.content

    if final_answer == "":
        final_answer = "Rate limit exceeded. Please reduce the message size or try again later."
    else:
        chat_context = summarize(chat_context, user_query, final_answer)

    # Update UI chat history
    chat_history.append({
        "role": "user",
        "content": f"{user_query}"
    })
    chat_history.append({
        "role": "assistant",
        "content": f"{final_answer}"
    })

    return chat_history, json.dumps(tool_calls_collected, indent=2), reasonings_collected, "", chat_context


def reset_chat():
    return [], "{}", "", "", [], ""

# -----------------------
# Gradio UI
# -----------------------

with gr.Blocks(title="ReAct AI Chatbot") as demo:
    gr.Markdown("## ReAct AI Chatbot\nMulti-Turn • Tool-Aware")

    chat_state = gr.State([])
    chat_context = gr.State("")


    with gr.Row(height="calc(100vh - 350px)"):
        chatbot = gr.Chatbot(
            label="Conversation",
            scale=3,
            height="calc(100vh - 350px)",
        )

        with gr.Column(scale=1):
            tool_calls_box = gr.Code(
                label="AI Tool Calls (Structured JSON)",
                language="json",
                max_lines=11,
                lines=11,
            )

            reasoning_box = gr.Code(
                label="Model Reasoning",
                language="markdown",
                max_lines=7,
                lines=7
            )

    user_input = gr.Textbox(
        label="Your Message",
        placeholder="Ask something...",
        lines=2
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        reset_btn = gr.Button("Reset Chat")

    user_input.submit(
        fn=chat,
        inputs=[user_input, chat_state, chat_context],
        outputs=[chatbot, tool_calls_box, reasoning_box, user_input, chat_context]
    )

    send_btn.click(
        fn=chat,
        inputs=[user_input, chat_state, chat_context],
        outputs=[chatbot, tool_calls_box, reasoning_box, user_input, chat_context]
    )

    reset_btn.click(
        fn=reset_chat,
        outputs=[chatbot, tool_calls_box, reasoning_box, user_input, chat_state, chat_context]
    )

# -----------------------
# Run
# -----------------------

if __name__ == "__main__":
    demo.launch()
