"""
Загрузка исторических данных о погоде через Meteostat
"""
from datetime import datetime
import pandas as pd
import os

# Альтернативный импорт
try:
    from meteostat import Point, Hourly
    print("✅ Импорт метода 1 успешен")
except ImportError:
    try:
        from meteostat.core.point import Point
        from meteostat.core.hourly import Hourly
        print("✅ Импорт метода 2 успешен")
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("Попробуй: pip uninstall meteostat && pip install meteostat==1.6.8")
        exit(1)

# Координаты Астаны
astana = Point(51.1694, 71.4491, 347)

# Период данных (последние 2 года)
start = datetime(2024, 1, 1)
end = datetime(2026, 1, 23)

print("🌍 Загрузка данных о погоде в Астане...")
print(f"📆 Период: {start.date()} — {end.date()}")

# Получить почасовые данные
data = Hourly(astana, start, end)
df = data.fetch()

# Проверка, что данные загружены
if df.empty:
    print("⚠️ Данные не загружены! Попробуй изменить период или координаты.")
    exit(1)

# Переименовать колонки
df = df.rename(columns={
    'temp': 'temperature',
    'dwpt': 'dew_point',
    'rhum': 'humidity',
    'prcp': 'precipitation',
    'snow': 'snow',
    'wdir': 'wind_deg',
    'wspd': 'wind_speed',
    'wpgt': 'wind_gust',
    'pres': 'pressure',
    'tsun': 'sunshine',
    'coco': 'weather_code'
})

# Сбросить индекс (timestamp станет колонкой)
df.reset_index(inplace=True)

# Создать папку если не существует
os.makedirs('data/raw', exist_ok=True)

# Сохранить в CSV
filepath = 'data/raw/astana_historical.csv'
df.to_csv(filepath, index=False)

print(f"\n✅ Данные успешно загружены!")
print(f"📊 Всего записей: {len(df)}")
print(f"💾 Размер файла: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
print(f"📁 Файл сохранён: {filepath}")

# Показать статистику
print(f"\n📈 Статистика по температуре:")
print(f"   Минимум: {df['temperature'].min():.1f}°C")
print(f"   Максимум: {df['temperature'].max():.1f}°C")
print(f"   Среднее: {df['temperature'].mean():.1f}°C")

print(f"\n📋 Первые 5 записей:")
print(df.head())

print(f"\n📋 Колонки в датасете:")
print(df.columns.tolist())
