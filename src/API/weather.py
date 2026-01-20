import os
from langchain_community.utilities import OpenWeatherMapAPIWrapper

def curr_weather(city: str):
    weather = OpenWeatherMapAPIWrapper()
    weather_data = weather.run(city)
    return weather_data