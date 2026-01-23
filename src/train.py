"""
Training pipeline для LSTM модели
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import os
from datetime import datetime
import json

from models.lstm_model import WeatherLSTM

class Trainer:
    """Класс для обучения модели"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        batch_size: int = 64
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, 
        )
        
        # История обучения
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    def load_data(self, data_dir: str = 'data/processed'):
        """Загрузить обработанные данные"""
        print("📂 Загрузка данных...")
        
        X_train = np.load(f'{data_dir}/X_train.npy')
        y_train = np.load(f'{data_dir}/y_train.npy')
        X_val = np.load(f'{data_dir}/X_val.npy')
        y_val = np.load(f'{data_dir}/y_val.npy')
        
        # Конвертировать в PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).unsqueeze(1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Создать DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        
        print(f"✅ Данные загружены:")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print(f"   Device: {self.device}")
        
    def train_epoch(self) -> float:
        """Один epoch обучения"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Валидация"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int = 50, save_dir: str = 'models'):
        """Полный цикл обучения"""
        print(f"\n🚀 Начало обучения ({num_epochs} epochs)...\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Сохранить историю
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Вывод
            print(f"Epoch {epoch:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Сохранить лучшую модель
                self.save_model(f'{save_dir}/best_model.pth')
                print(f"   ✅ Лучшая модель сохранена (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\n⚠️  Early stopping на epoch {epoch}")
                    break
        
        print(f"\n✅ Обучение завершено!")
        print(f"   Лучший val_loss: {best_val_loss:.6f}")
        
        # Сохранить историю
        self.save_history(f'{save_dir}/training_history.json')
        
    def save_model(self, filepath: str):
        """Сохранить модель"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
    
    def save_history(self, filepath: str):
        """Сохранить историю обучения"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

# Запуск обучения
if __name__ == "__main__":
    print("🧠 Инициализация обучения LSTM модели...\n")
    
    # Создать модель
    model = WeatherLSTM(
        input_size=10,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False
    )
    
    # Создать trainer
    trainer = Trainer(
        model=model,
        learning_rate=0.001,
        batch_size=64
    )
    
    # Загрузить данные
    trainer.load_data('data/processed')
    
    # Обучить модель
    trainer.train(num_epochs=50, save_dir='models')
    
    print("\n🎉 Готово! Модель сохранена в models/best_model.pth")
