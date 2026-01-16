# # Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.model import get_model
from src.tools import get_tools

# -----------------------
# ReAct Agent (SUPPORTED)
# -----------------------

llm = get_model()
tools = get_tools()
agent = create_agent(
    llm,
    tools=tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)

# -----------------------
# Run loop
# -----------------------

if __name__ == "__main__":
    while True:
        query = input("\nAsk something (or 'exit'): ")
        if query.lower() == "exit":
            break

        human_msg = HumanMessage(query)
        messages = {"messages": [human_msg]}

        response = agent.invoke(messages)

        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                print("AI requested tool:")
                print(msg.tool_calls)

            elif isinstance(msg, ToolMessage):
                print("Tool returned:")
                print(msg.content)

            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                print("Final AI response:")
                print(msg.content)


