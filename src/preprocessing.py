"""
Preprocessing данных для обучения модели прогнозирования погоды
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import pickle
import os

class WeatherPreprocessor:
    """Класс для предобработки данных о погоде"""
    
    def __init__(self, sequence_length: int = 24):
        """
        Args:
            sequence_length: Длина временного окна (часов) для LSTM
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns: list[str] = []   # populated by select_features()
        self.target_column = 'temperature'
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загрузить данные из CSV"""
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Выбрать признаки для модели"""
        # Основные признаки
        self.feature_columns = [
            'temperature',
            'humidity',
            'pressure',
            'wind_speed',
            'dew_point'
        ]
        
        # Добавить временные признаки
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        
        # Циклические признаки (для часов и месяцев)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Добавить временные признаки к списку
        self.feature_columns += [
            'hour_sin', 'hour_cos',
            'month_sin', 'month_cos',
            'day_of_week'
        ]
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений"""
        # Интерполяция для числовых колонок
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        # Удалить оставшиеся NaN
        df = df.dropna(subset=self.feature_columns)
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Нормализация данных в диапазон [0, 1]"""
        if fit:
            df[self.feature_columns] = self.scaler.fit_transform(df[self.feature_columns])
        else:
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создать временные последовательности для LSTM
        
        Args:
            data: Массив признаков [samples, features]
            target: Массив целевой переменной [samples]
            
        Returns:
            X: [samples, sequence_length, features]
            y: [samples]
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7, 
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделить данные на train/val/test
        
        Args:
            train_ratio: Доля обучающей выборки
            val_ratio: Доля валидационной выборки
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df[:train_end].copy()
        val_df = df[train_end:val_end].copy()
        test_df = df[val_end:].copy()
        
        print("[*] Splitting data:")
        print(f"   Train: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
        print(f"   Val:   {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
        print(f"   Test:  {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def preprocess(self, filepath: str, save_dir: str = 'data/processed') -> dict:
        """
        Полный pipeline preprocessing
        
        Returns:
            dict с обработанными данными
        """
        print("[*] Starting preprocessing...")

        # 1. Load
        print("   1/6 Loading data...")
        df = self.load_data(filepath)
        print(f"       Loaded: {len(df)} records")

        # 2. Feature Engineering
        print("   2/6 Feature Engineering...")
        df = self.select_features(df)
        print(f"       Features: {len(self.feature_columns)}")

        # 3. Missing values
        print("   3/6 Handling missing values...")
        df = self.handle_missing_values(df)
        print(f"       Remaining: {len(df)} records")

        # 4. Split
        print("   4/6 Splitting data...")
        train_df, val_df, test_df = self.split_data(df)

        # 5. Normalise (fit on train only!)
        print("   5/6 Normalizing...")
        train_df = self.normalize_data(train_df, fit=True)
        val_df = self.normalize_data(val_df, fit=False)
        test_df = self.normalize_data(test_df, fit=False)
        
        # 6. Создание последовательностей
        print("   6/6 Создание последовательностей...")
        
        X_train, y_train = self.create_sequences(
            train_df[self.feature_columns].values,
            train_df[self.target_column].values
        )
        X_val, y_val = self.create_sequences(
            val_df[self.feature_columns].values,
            val_df[self.target_column].values
        )
        X_test, y_test = self.create_sequences(
            test_df[self.feature_columns].values,
            test_df[self.target_column].values
        )
        
        print(f"       X_train shape: {X_train.shape}")
        print(f"       X_val shape:   {X_val.shape}")
        print(f"       X_test shape:  {X_test.shape}")
        
        # 7. Save
        print("   Saving processed data...")
        os.makedirs(save_dir, exist_ok=True)
        
        np.save(f'{save_dir}/X_train.npy', X_train)
        np.save(f'{save_dir}/y_train.npy', y_train)
        np.save(f'{save_dir}/X_val.npy', X_val)
        np.save(f'{save_dir}/y_val.npy', y_val)
        np.save(f'{save_dir}/X_test.npy', X_test)
        np.save(f'{save_dir}/y_test.npy', y_test)
        
        # Сохранить scaler
        with open(f'{save_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Сохранить метаданные
        metadata = {
            'sequence_length': self.sequence_length,
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns),
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
        
        with open(f'{save_dir}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n[+] Preprocessing complete!")
        print(f"   Files saved to: {save_dir}/")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'metadata': metadata
        }

# Тестирование
if __name__ == "__main__":
    preprocessor = WeatherPreprocessor(sequence_length=24)
    
    data = preprocessor.preprocess(
        filepath='data/raw/astana_historical.csv',
        save_dir='data/processed'
    )
    
    print("\n[*] Summary:")
    print(f"   Input shape:     {data['X_train'].shape}")
    print(f"   Target shape:    {data['y_train'].shape}")
    print(f"   Features:        {data['metadata']['n_features']}")
    print(f"   Sequence length: {data['metadata']['sequence_length']} hours")
