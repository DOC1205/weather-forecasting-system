"""
Training pipeline for the Weather Forecasting System (Astana).

Supports two model architectures selectable via --model flag:
  lstm   – baseline WeatherLSTM (2-layer LSTM + FC head, ~87k params)
  hybrid – HybridWeatherModel (LSTM + TCN + Transformer + gated fusion, ~340k params)

Usage from project root:
    python src/train.py --model hybrid --epochs 50
    python src/train.py --model lstm   --epochs 50
"""

import sys
import os

# Allow imports of sibling modules (models/) whether the script is invoked
# from the project root or from inside src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models.lstm_model import WeatherLSTM
from models.hybrid_model import HybridWeatherModel, count_parameters


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """
    Model-agnostic training loop for WeatherLSTM and HybridWeatherModel.

    Accepts any nn.Module that maps [batch, seq_len, features] → [batch, 1].
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        model_config: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        batch_size: int = 64,
    ):
        """
        Args:
            model:        Instantiated and un-trained nn.Module.
            model_type:   'lstm' or 'hybrid' — stored in the checkpoint so that
                          the app and evaluator can reconstruct the model without
                          guessing.
            model_config: Dict of constructor kwargs used to build the model.
                          Stored in the checkpoint for full reproducibility.
            device:       'cuda' or 'cpu'.
            learning_rate: Initial Adam learning rate.
            batch_size:   Mini-batch size for train/val DataLoaders.
        """
        self.model        = model.to(device)
        self.model_type   = model_type
        self.model_config = model_config
        self.device       = device
        self.learning_rate = learning_rate
        self.batch_size   = batch_size

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        self.history: dict = {"train_loss": [], "val_loss": [], "learning_rate": []}

        # Declared here so Pylance knows these exist; populated by load_data()
        self.train_loader: DataLoader  # type: ignore[assignment]
        self.val_loader:   DataLoader  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, data_dir: str = "data/processed") -> None:
        """
        Load preprocessed .npy arrays and wrap them in PyTorch DataLoaders.

        Args:
            data_dir: Directory produced by src/preprocessing.py containing
                      X_train.npy, y_train.npy, X_val.npy, y_val.npy.
        """
        print("[*] Loading data...")

        X_train = torch.FloatTensor(np.load(f"{data_dir}/X_train.npy"))
        y_train = torch.FloatTensor(np.load(f"{data_dir}/y_train.npy")).unsqueeze(1)
        X_val   = torch.FloatTensor(np.load(f"{data_dir}/X_val.npy"))
        y_val   = torch.FloatTensor(np.load(f"{data_dir}/y_val.npy")).unsqueeze(1)

        self.train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True, num_workers=0,
        )
        self.val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

        print(f"[+] Data loaded:")
        print(f"   Train: {len(X_train):,} samples ({len(self.train_loader)} batches)")
        print(f"   Val:   {len(X_val):,} samples ({len(self.val_loader)} batches)")
        print(f"   Device: {self.device}")

    # ------------------------------------------------------------------
    # Single epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self) -> float:
        """
        Run one training epoch over the entire training DataLoader.

        Applies gradient clipping (max_norm=1.0) to prevent exploding
        gradients — especially important for the LSTM branch.

        Returns:
            Mean training loss for this epoch.
        """
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in self.train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self) -> float:
        """
        Compute validation loss without updating model weights.

        Returns:
            Mean validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                total_loss += self.criterion(self.model(X_batch), y_batch).item()

        return total_loss / len(self.val_loader)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, num_epochs: int = 50, save_dir: str = "models") -> None:
        """
        Full training loop with early stopping and best-model checkpointing.

        Saves:
          - <save_dir>/best_model.pth        — best checkpoint (lowest val_loss)
          - <save_dir>/training_history.json — loss & LR curves

        Args:
            num_epochs: Maximum number of epochs (early stopping may stop earlier).
            save_dir:   Directory for output artefacts.
        """
        print(f"\n[*] Training model '{self.model_type}' ({num_epochs} epochs)...")
        print(f"   Parameters: {count_parameters(self.model):,}")
        print(f"   Device:     {self.device}\n")

        best_val_loss    = float("inf")
        patience_counter = 0
        max_patience     = 10
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            train_loss = self._train_epoch()
            val_loss   = self._validate()
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.scheduler.step(val_loss)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(current_lr)

            print(
                f"Epoch {epoch:3d}/{num_epochs}  |  "
                f"Train: {train_loss:.6f}  |  "
                f"Val: {val_loss:.6f}  |  "
                f"LR: {current_lr:.2e}"
            )

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                patience_counter = 0
                self._save_checkpoint(f"{save_dir}/best_model.pth")
                print(f"   [+] Best model saved (val_loss={val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"\n[!] Early stopping at epoch {epoch} (patience={max_patience})")
                    break

        print(f"\n[+] Training complete! Best val_loss: {best_val_loss:.6f}")
        self._save_history(f"{save_dir}/training_history.json")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filepath: str) -> None:
        """
        Save a full checkpoint including model weights, optimizer state,
        training history, model type and constructor config.

        The extra keys allow load_model() in app.py and ModelEvaluator in
        evaluate.py to reconstruct the exact architecture without guessing.

        Args:
            filepath: Destination path, e.g. 'models/best_model.pth'.
        """
        torch.save(
            {
                "model_state_dict":   self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history":            self.history,
                "model_type":         self.model_type,   # 'lstm' | 'hybrid'
                "model_config":       self.model_config, # constructor kwargs
                "saved_at":           datetime.now().isoformat(),
            },
            filepath,
        )

    def _save_history(self, filepath: str) -> None:
        """
        Persist the training history dict as JSON.

        Args:
            filepath: Destination path, e.g. 'models/training_history.json'.
        """
        with open(filepath, "w") as fh:
            json.dump(self.history, fh, indent=2)
        print(f"[+] Training history saved: {filepath}")


# ============================================================================
# Model factory
# ============================================================================

def build_model(model_type: str, n_features: int) -> tuple[nn.Module, dict]:
    """
    Instantiate a model and return it together with its config dict.

    Args:
        model_type: 'lstm' or 'hybrid'.
        n_features: Number of input features (10 for this project).

    Returns:
        Tuple (model: nn.Module, config: dict)
        where config holds the constructor kwargs used — stored in checkpoint.

    Raises:
        ValueError: If model_type is not recognised.
    """
    n = int(n_features)  # ensure int — metadata.get() returns Any

    if model_type == "lstm":
        # Store config separately; construct model with explicit typed args
        config: dict = {"input_size": n, "hidden_size": 128, "num_layers": 2, "dropout": 0.2}
        model: nn.Module = WeatherLSTM(input_size=n, hidden_size=128, num_layers=2, dropout=0.2)

    elif model_type == "hybrid":
        config = {
            "input_size": n, "lstm_hidden": 128, "lstm_layers": 2,
            "tcn_channels": 64, "tcn_levels": 4, "transformer_d_model": 64,
            "transformer_heads": 4, "transformer_layers": 2, "dropout": 0.2,
        }
        model = HybridWeatherModel(
            input_size=n, lstm_hidden=128, lstm_layers=2,
            tcn_channels=64, tcn_levels=4, transformer_d_model=64,
            transformer_heads=4, transformer_layers=2, dropout=0.2,
        )

    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose 'lstm' or 'hybrid'.")

    return model, config


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the weather forecasting model for Astana."
    )
    parser.add_argument(
        "--model",
        choices=["lstm", "hybrid"],
        default="hybrid",
        help="Model architecture to train (default: hybrid)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Maximum number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Initial Adam learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, dest="batch_size",
        help="Mini-batch size (default: 64)",
    )
    parser.add_argument(
        "--data-dir", default="data/processed", dest="data_dir",
        help="Directory with preprocessed .npy files (default: data/processed)",
    )
    parser.add_argument(
        "--save-dir", default="models", dest="save_dir",
        help="Directory for saving checkpoints and history (default: models)",
    )
    args = parser.parse_args()

    # Load n_features from metadata
    meta_path = os.path.join(args.data_dir, "metadata.pkl")
    if not os.path.exists(meta_path):
        print(f"[!] metadata.pkl not found at {meta_path}")
        print("   Run: python src/preprocessing.py")
        sys.exit(1)

    with open(meta_path, "rb") as fh:
        metadata = pickle.load(fh)
    n_features = metadata.get("n_features", 10)

    print(f"\n[*] Initializing model '{args.model}' (n_features={n_features})...")
    model, config = build_model(args.model, n_features)
    print(f"   Parameters: {count_parameters(model):,}")

    trainer = Trainer(
        model=model,
        model_type=args.model,
        model_config=config,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    trainer.load_data(args.data_dir)
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    print(f"\n[+] Done! Model saved to {args.save_dir}/best_model.pth")
