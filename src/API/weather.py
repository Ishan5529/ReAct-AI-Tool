# import requests
# import os

# API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

# def curr_weather(city: str):
#     api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
#     response = requests.get(api).json()
#     return response

import os
from langchain_community.utilities import OpenWeatherMapAPIWrapper

def curr_weather(city: str):
    weather = OpenWeatherMapAPIWrapper()
    weather_data = weather.run(city)
    return weather_data