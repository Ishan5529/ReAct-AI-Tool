import os
from tavily import TavilyClient

TAVILY_CLIENT_KEY = os.getenv("TAVILY_CLIENT_KEY")
def search_tavily(query, topic="general", time_range="day"):
    client = TavilyClient(TAVILY_CLIENT_KEY)
    response = client.search(
        query=query.lower(),
        include_answer="advanced",
        topic=topic.lower(),
        search_depth="advanced",
        max_results=6,
        time_range=time_range.lower()
    )
    return response