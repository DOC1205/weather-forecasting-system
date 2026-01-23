"""
LSTM модель для прогнозирования температуры
"""
import torch
import torch.nn as nn
from typing import Tuple

class WeatherLSTM(nn.Module):
    """LSTM модель для прогнозирования погоды"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Args:
            input_size: Количество признаков (10)
            hidden_size: Размер скрытого слоя LSTM
            num_layers: Количество слоёв LSTM
            dropout: Dropout для регуляризации
            bidirectional: Использовать двунаправленный LSTM
        """
        super(WeatherLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM слои
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully connected слои
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: [batch_size, sequence_length, input_size]
            
        Returns:
            output: [batch_size, 1]
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Взять последний выход
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            out = h_n[-1]
        
        # Fully connected
        output = self.fc(out)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Предсказание без градиентов"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

# Тестирование модели
if __name__ == "__main__":
    print("🧪 Тестирование LSTM модели...")
    
    # Параметры
    batch_size = 32
    sequence_length = 24
    input_size = 10
    
    # Создать модель
    model = WeatherLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=False
    )
    
    # Тестовые данные
    x = torch.randn(batch_size, sequence_length, input_size)
    
    # Forward pass
    output = model(x)
    
    print(f"✅ Модель создана успешно!")
    print(f"   Входная форма: {x.shape}")
    print(f"   Выходная форма: {output.shape}")
    print(f"\n📊 Параметры модели:")
    print(f"   Всего параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n🏗️  Архитектура модели:")
    print(model)
