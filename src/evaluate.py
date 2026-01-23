"""
Evaluation скрипт для оценки качества модели
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import json
import os

from models.lstm_model import WeatherLSTM

class ModelEvaluator:
    """Класс для оценки модели"""
    
    def __init__(self, model_path: str, data_dir: str = 'data/processed'):
        """
        Args:
            model_path: Путь к обученной модели
            data_dir: Директория с обработанными данными
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = data_dir
        
        # Загрузить метаданные
        with open(f'{data_dir}/metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Загрузить scaler
        with open(f'{data_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Создать и загрузить модель
        self.model = WeatherLSTM(
            input_size=self.metadata['n_features'],
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=False
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Модель загружена с {self.device}")
        
    def load_test_data(self):
        """Загрузить тестовые данные"""
        X_test = np.load(f'{self.data_dir}/X_test.npy')
        y_test = np.load(f'{self.data_dir}/y_test.npy')
        
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = y_test
        
        print(f"📊 Тестовые данные загружены: {len(self.y_test)} примеров")
        
    def predict(self):
        """Сделать предсказания на тестовых данных"""
        print("🔮 Генерация предсказаний...")
        
        with torch.no_grad():
            predictions = self.model(self.X_test).cpu().numpy().flatten()
        
        self.predictions = predictions
        
        # Денормализация (обратное преобразование)
        temp_index = self.scaler.feature_names_in_.tolist().index('temperature')
        
        # Создать dummy массив для обратного преобразования
        dummy = np.zeros((len(predictions), len(self.scaler.feature_names_in_)))
        dummy[:, temp_index] = predictions
        predictions_denorm = self.scaler.inverse_transform(dummy)[:, temp_index]
        
        dummy_true = np.zeros((len(self.y_test), len(self.scaler.feature_names_in_)))
        dummy_true[:, temp_index] = self.y_test
        y_test_denorm = self.scaler.inverse_transform(dummy_true)[:, temp_index]
        
        self.predictions_denorm = predictions_denorm
        self.y_test_denorm = y_test_denorm
        
        print(f"✅ Предсказания готовы")
        
    def calculate_metrics(self):
        """Вычислить метрики качества"""
        print("\n📈 Метрики качества:")
        
        # На нормализованных данных
        mae_norm = mean_absolute_error(self.y_test, self.predictions)
        rmse_norm = np.sqrt(mean_squared_error(self.y_test, self.predictions))
        r2_norm = r2_score(self.y_test, self.predictions)
        
        # На денормализованных данных (реальные температуры)
        mae = mean_absolute_error(self.y_test_denorm, self.predictions_denorm)
        rmse = np.sqrt(mean_squared_error(self.y_test_denorm, self.predictions_denorm))
        r2 = r2_score(self.y_test_denorm, self.predictions_denorm)
        mape = np.mean(np.abs((self.y_test_denorm - self.predictions_denorm) / self.y_test_denorm)) * 100
        
        print(f"\n🌡️  Реальные температуры:")
        print(f"   MAE:  {mae:.2f}°C")
        print(f"   RMSE: {rmse:.2f}°C")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        self.metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'mae_norm': mae_norm,
            'rmse_norm': rmse_norm,
            'r2_norm': r2_norm
        }
        
        return self.metrics
    
    def plot_predictions(self, save_path: str = 'docs/predictions.png', n_samples: int = 500):
        """График предсказаний vs реальных значений"""
        plt.figure(figsize=(15, 10))
        
        # График 1: Time series (первые n_samples)
        plt.subplot(2, 2, 1)
        indices = range(min(n_samples, len(self.y_test_denorm)))
        plt.plot(indices, self.y_test_denorm[:n_samples], label='Реальные', linewidth=2, alpha=0.7)
        plt.plot(indices, self.predictions_denorm[:n_samples], label='Предсказания', linewidth=2, alpha=0.7)
        plt.xlabel('Индекс времени')
        plt.ylabel('Температура (°C)')
        plt.title('Предсказания vs Реальные значения', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График 2: Scatter plot
        plt.subplot(2, 2, 2)
        plt.scatter(self.y_test_denorm, self.predictions_denorm, alpha=0.5, s=10)
        plt.plot([self.y_test_denorm.min(), self.y_test_denorm.max()], 
                 [self.y_test_denorm.min(), self.y_test_denorm.max()], 
                 'r--', linewidth=2, label='Идеальная линия')
        plt.xlabel('Реальная температура (°C)')
        plt.ylabel('Предсказанная температура (°C)')
        plt.title(f'Correlation (R² = {self.metrics["r2"]:.4f})', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График 3: Residuals (ошибки)
        plt.subplot(2, 2, 3)
        residuals = self.y_test_denorm - self.predictions_denorm
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Ошибка предсказания (°C)')
        plt.ylabel('Частота')
        plt.title(f'Распределение ошибок (MAE = {self.metrics["mae"]:.2f}°C)', fontweight='bold')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
        plt.grid(True, alpha=0.3)
        
        # График 4: Residuals vs predictions
        plt.subplot(2, 2, 4)
        plt.scatter(self.predictions_denorm, residuals, alpha=0.5, s=10)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Предсказанная температура (°C)')
        plt.ylabel('Ошибка (°C)')
        plt.title('Ошибки vs Предсказания', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ График сохранён: {save_path}")
    
    def plot_training_history(self, save_path: str = 'docs/training_history.png'):
        """График истории обучения"""
        history_path = 'models/training_history.json'
        
        if not os.path.exists(history_path):
            print("⚠️  История обучения не найдена")
            return
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        plt.figure(figsize=(15, 5))
        
        # График 1: Loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('История обучения: Loss', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # График 2: Learning Rate
        plt.subplot(1, 2, 2)
        plt.plot(history['learning_rate'], linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule', fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ График истории обучения сохранён: {save_path}")
    
    def generate_report(self, save_path: str = 'docs/evaluation_report.txt'):
        """Сгенерировать текстовый отчёт"""
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          ОТЧЁТ ОБ ОЦЕНКЕ МОДЕЛИ ПРОГНОЗИРОВАНИЯ ПОГОДЫ       ║
╚══════════════════════════════════════════════════════════════╝

📅 Дата оценки: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 ДАННЫЕ:
   • Тестовых примеров: {len(self.y_test)}
   • Признаков: {self.metadata['n_features']}
   • Длина последовательности: {self.metadata['sequence_length']} часов

🧠 МОДЕЛЬ:
   • Архитектура: LSTM
   • Hidden size: 128
   • Layers: 2
   • Parameters: ~87,000

📈 МЕТРИКИ КАЧЕСТВА:

   🌡️  Температура (денормализованная):
      • MAE:  {self.metrics['mae']:.2f}°C
      • RMSE: {self.metrics['rmse']:.2f}°C
      • R²:   {self.metrics['r2']:.4f}
      • MAPE: {self.metrics['mape']:.2f}%

   📊 Нормализованные значения:
      • MAE:  {self.metrics['mae_norm']:.6f}
      • RMSE: {self.metrics['rmse_norm']:.6f}
      • R²:   {self.metrics['r2_norm']:.4f}

💡 ИНТЕРПРЕТАЦИЯ:

   • R² = {self.metrics['r2']:.4f}: {'Отличная' if self.metrics['r2'] > 0.9 else 'Хорошая' if self.metrics['r2'] > 0.8 else 'Удовлетворительная'} корреляция
   • MAE = {self.metrics['mae']:.2f}°C: В среднем ошибка ±{self.metrics['mae']:.2f}°C
   • RMSE = {self.metrics['rmse']:.2f}°C: Учитывает большие отклонения

✅ ЗАКЛЮЧЕНИЕ:
   Модель {'успешно' if self.metrics['mae'] < 3.0 else 'удовлетворительно'} прогнозирует температуру
   со средней ошибкой {self.metrics['mae']:.2f}°C на горизонте {self.metadata['sequence_length']} часов.

"""
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"✅ Отчёт сохранён: {save_path}")

# Запуск evaluation
if __name__ == "__main__":
    print("📊 EVALUATION: Оценка модели\n")
    
    # Создать evaluator
    evaluator = ModelEvaluator(
        model_path='models/best_model.pth',
        data_dir='data/processed'
    )
    
    # Загрузить тестовые данные
    evaluator.load_test_data()
    
    # Сделать предсказания
    evaluator.predict()
    
    # Вычислить метрики
    evaluator.calculate_metrics()
    
    # Построить графики
    evaluator.plot_predictions(save_path='docs/predictions.png')
    evaluator.plot_training_history(save_path='docs/training_history.png')
    
    # Сгенерировать отчёт
    evaluator.generate_report(save_path='docs/evaluation_report.txt')
    
    print("\n🎉 Evaluation завершён!")
