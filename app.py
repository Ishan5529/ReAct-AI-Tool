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
    system_prompt=SystemMessage("""You are a reasoning assistant. You must decide whether the question can be answered confidently using general knowledge, or whether external information is required.
        
        Rules:
        1) Answer directly when the question is simple, factual, or commonly known.
        2) Respond to the greetings, if present in the query.
        3) Use weather_search_tool, only for finding current meteorological data of a city and return summarised output.
        4) Use the web_search_tool only when the answer depends on recent, dynamic, or unverifiable information.
        5) If a meteorological question cannot be fully or confidently answered using only the output of weather_search_tool, then perform a follow-up call to web_search_tool to obtain the missing or clarifying information.
        6) When using a tool, call exactly one tool at a time and provide only the required arguments defined by the tool schema.
        7) After a tool call, read the tool's output and produce a detailed and human-readable final answer.
        8) Do not mention tools, function calls, or internal reasoning in the final answer.
        """)
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
            # break
            if isinstance(msg, AIMessage) and msg.tool_calls:
                print("\nAI requested tool:")
                print(msg.tool_calls)

            # elif isinstance(msg, ToolMessage):
            #     print("Tool returned:")
            #     print(msg.content)

            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                print("\nFinal AI response:")
                print(msg.content)


