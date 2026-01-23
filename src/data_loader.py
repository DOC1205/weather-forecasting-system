"""
Загрузка данных о погоде из OpenWeatherMap API
"""
import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Загрузить переменные окружения из .env
load_dotenv()

class WeatherDataLoader:
    """Класс для загрузки данных о погоде"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5"
        
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY не найден в .env файле!")
    
    def get_current_weather(self, city: str) -> dict:
        """Получить текущую погоду для города"""
        url = f"{self.base_url}/weather"
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric',  # Цельсий
            'lang': 'ru'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_forecast(self, city: str, days: int = 5) -> pd.DataFrame:
        """Получить прогноз погоды на несколько дней"""
        url = f"{self.base_url}/forecast"
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric',
            'lang': 'ru'
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Преобразовать в DataFrame
        records = []
        for item in data['list']:
            records.append({
                'timestamp': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'feels_like': item['main']['feels_like'],
                'temp_min': item['main']['temp_min'],
                'temp_max': item['main']['temp_max'],
                'pressure': item['main']['pressure'],
                'humidity': item['main']['humidity'],
                'weather': item['weather'][0]['main'],
                'weather_description': item['weather'][0]['description'],
                'wind_speed': item['wind']['speed'],
                'wind_deg': item['wind']['deg'],
                'clouds': item['clouds']['all']
            })
        
        df = pd.DataFrame(records)
        return df

# Тестирование
if __name__ == "__main__":
    print("🌍 Загрузка данных о погоде...")
    loader = WeatherDataLoader()
    
    # Получить текущую погоду в Астане
    print("\n🌤️  Текущая погода в Астане:")
    current = loader.get_current_weather("Astana")
    print(f"   Температура: {current['main']['temp']}°C")
    print(f"   Ощущается как: {current['main']['feels_like']}°C")
    print(f"   Влажность: {current['main']['humidity']}%")
    print(f"   Давление: {current['main']['pressure']} hPa")
    print(f"   Ветер: {current['wind']['speed']} м/с")
    print(f"   Описание: {current['weather'][0]['description']}")
    
    # Получить прогноз на 5 дней
    print("\n📊 Загрузка прогноза на 5 дней...")
    forecast = loader.get_forecast("Astana")
    print(f"   Получено {len(forecast)} записей")
    print("\nПервые 10 записей:")
    print(forecast.head(10)[['timestamp', 'temperature', 'humidity', 'weather_description']])
    
    # Сохранить в CSV
    os.makedirs('data/raw', exist_ok=True)
    forecast.to_csv('data/raw/astana_forecast.csv', index=False)
    print(f"\n✅ Данные сохранены в data/raw/astana_forecast.csv")
    print(f"   Размер файла: {os.path.getsize('data/raw/astana_forecast.csv')} байт")
