"""
Скрипт для ежедневного сбора данных о погоде
Запускается автоматически или вручную
"""
import os
import sys
import pandas as pd
from datetime import datetime

# Добавить путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import WeatherDataLoader

def collect_weather_data():
    """Собрать текущую погоду и добавить в CSV"""
    
    loader = WeatherDataLoader()
    
    # Получить текущую погоду
    current = loader.get_current_weather("Astana")
    
    # Создать запись
    new_row = {
        'timestamp': datetime.now(),
        'temperature': current['main']['temp'],
        'feels_like': current['main']['feels_like'],
        'temp_min': current['main']['temp_min'],
        'temp_max': current['main']['temp_max'],
        'pressure': current['main']['pressure'],
        'humidity': current['main']['humidity'],
        'weather': current['weather'][0]['main'],
        'weather_description': current['weather'][0]['description'],
        'wind_speed': current['wind']['speed'],
        'wind_deg': current['wind'].get('deg', 0),
        'clouds': current['clouds']['all']
    }
    
    # Путь к файлу
    filepath = 'data/raw/astana_historical.csv'
    
    # Создать DataFrame
    df = pd.DataFrame([new_row])
    
    # Проверить, существует ли файл
    if os.path.exists(filepath):
        # Добавить к существующим данным
        df.to_csv(filepath, mode='a', header=False, index=False)
        print(f"✅ Данные добавлены к существующему файлу")
    else:
        # Создать новый файл
        df.to_csv(filepath, mode='w', header=True, index=False)
        print(f"✅ Создан новый файл с данными")
    
    print(f"📅 Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌡️  Температура: {new_row['temperature']}°C")
    print(f"💧 Влажность: {new_row['humidity']}%")
    print(f"📊 Всего записей в файле: {len(pd.read_csv(filepath))}")

if __name__ == "__main__":
    collect_weather_data()
