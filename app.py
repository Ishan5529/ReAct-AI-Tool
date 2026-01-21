# # Load environment variables
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from src.model import get_model
from src.tools import get_tools
from datetime import datetime

# -----------------------
# Helper Function
# -----------------------

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

# -----------------------
# ReAct Agent (SUPPORTED)
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
        5) ALWAYS USE real_time_tool to determine the current time before answering any question or calling tools that depends on, references, or compares against the present / latest time.
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
# Run loop
# -----------------------

if __name__ == "__main__":
    chat_history = []
    while True:
        query = input("\nAsk something (or 'exit'): ")
        if query.lower() == "exit":
            break
        
        human_msg = HumanMessage(query)
        chat_history.append(human_msg)
        messages = {"messages": chat_history[-10:]}

        response = agent.invoke(messages)

        for msg in response["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                print("\nAI requested tool:")
                print(msg.tool_calls)

            elif isinstance(msg, AIMessage) and not msg.tool_calls:
                final_msg = msg.content
        print("\nFinal AI response:")
        print(final_msg)
        chat_history.append(AIMessage(final_msg[:500]))


