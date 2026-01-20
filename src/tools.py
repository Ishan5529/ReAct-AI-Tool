from langchain.tools import tool
from src.API.web_search import search_tavily
from src.API.weather import curr_weather

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

@tool("weather_search_tool", description="Search weather conditions of a city. input -> city: str : output -> str")
def get_weather(city: str) -> str:
    """
    Search for current meteorological data using Open Weather Map.

    Args:
        city: Search City

    Returns:
        Search results as string
    """
    return curr_weather(city)

def get_tools():
    tools = [web_search, get_weather]
    return tools