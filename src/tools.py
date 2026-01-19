from langchain.tools import tool
from src.API.web_search import search_tavily

# -----------------------
# Tools
# -----------------------

@tool("web_search_tool", description="Search a query on web. Returns the search result. input -> query: str, topic: 'general' or 'news' or 'finance'}, time_range: 'day' or 'week' or 'month' or 'year' ; output -> dict")
def web_search(query: str, topic: str, time_range: str) -> str:
    """
    Search the web using Tavily.

    Args:
        query: Search query string
        topic: Topic category (general, news, finance)
        time_range: Time range filter (day, week, month, year)

    Returns:
        Search results as dictionary
    """
    return search_tavily(query, topic, time_range)

@tool("adder", description="Performs sum calculations. Use this for calculating sum. input -> int, int : output -> dict")
def add(a: int, b: int) -> int:
    return a + b

def get_tools():
    tools = [web_search, add]
    return tools