import requests
import os

API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

def curr_weather(city: str):
    api = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={API_KEY}"
    response = requests.get(api).json()
    return response