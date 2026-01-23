# weather-forecasting-system
Advanced weather forecasting with LSTM, TCN, Transformer
# 🌤️ Weather Forecasting System

**Advanced weather forecasting with LSTM, TCN, Transformer**

Система прогнозирования температуры воздуха с использованием глубокого обучения (LSTM) на основе исторических метеорологических данных.

---

## 📊 О проекте

Дипломная работа по созданию системы прогнозирования погоды с использованием современных архитектур нейронных сетей.

### Ключевые особенности

- ✅ **Сбор данных**: OpenWeatherMap API + Meteostat (18,073 записей за 2 года)
- ✅ **Feature Engineering**: 10 признаков (температура, влажность, давление, ветер, временные признаки)
- ✅ **Архитектура**: LSTM с 2 слоями, 128 hidden units
- ✅ **Метрики**: MAE ~1.8-2.5°C, RMSE ~2.4-3.2°C, R² > 0.95
- ✅ **Визуализация**: Графики EDA, предсказаний, истории обучения

---

## 🏗️ Структура проекта

weather-forecasting-system/
├── data/
│ ├── raw/ # Исходные данные
│ │ └── astana_historical.csv
│ ├── processed/ # Обработанные данные
│ │ ├── X_train.npy
│ │ ├── scaler.pkl
│ │ └── metadata.pkl
│ └── external/ # Внешние датасеты
├── src/
│ ├── data_loader.py # Загрузка данных из API
│ ├── preprocessing.py # Feature engineering
│ ├── models/
│ │ └── lstm_model.py # LSTM архитектура
│ ├── train.py # Training pipeline
│ └── evaluate.py # Evaluation & metrics
├── notebooks/
│ └── 01_eda_weather_data.ipynb # Exploratory Data Analysis
├── models/
│ ├── best_model.pth # Обученная модель
│ └── training_history.json # История обучения
├── docs/
│ ├── predictions.png # Графики предсказаний
│ ├── training_history.png # График обучения
│ ├── evaluation_report.txt # Отчёт об оценке
│ └── *.png # Графики из EDA
├── configs/ # Конфигурационные файлы
├── tests/ # Unit тесты
├── scripts/ # Утилиты и скрипты
├── requirements.txt # Зависимости Python
├── .env.example # Пример конфигурации
├── .gitignore
└── README.md
```bash
git clone https://github.com/DOC1205/weather-forecasting-system.git
cd weather-forecasting-system