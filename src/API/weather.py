import os
from langchain_community.utilities import OpenWeatherMapAPIWrapper

def curr_weather(city: str, country: str = "") -> str:
    weather = OpenWeatherMapAPIWrapper()
    try:
        weather_data = weather.run(f"{city},{country}")
    except Exception as e:
        weather_data = f"country({country}) and city ({city}) pair doesn't exists. Retry with a different pair."
    return weather_data