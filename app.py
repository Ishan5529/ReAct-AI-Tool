# # Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain.messages import HumanMessage, AIMessage
from src.agent import setup_agent
from rag_summarize import summarize

# -----------------------
# ReAct Agent (SUPPORTED)
# -----------------------
agent = setup_agent()

# -----------------------
# Run loop
# -----------------------

if __name__ == "__main__":
    context = ""
    while True:
        query = input("\nAsk something (or 'exit'): ")
        if query.lower() == "exit":
            break
        
        human_msg = HumanMessage(f"PREVIOUS CONTEXT: {context}\n\nCURRENT_QUERY: {query}")
        messages = {"messages": [human_msg]}

        response = agent.invoke(messages)
        context = summarize(context, query, response)

        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                print("\nAI requested tool:")
                print(msg.tool_calls)

            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                final_msg = msg.content
        print("\nFinal AI response:")
        print(final_msg)


