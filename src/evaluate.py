"""
Evaluation script for the Weather Forecasting System (Astana).

Loads the best checkpoint produced by train.py, runs inference on the held-out
test set, computes all metrics, saves visualisations and a JSON metrics file
that the Streamlit app reads directly.

Supports both WeatherLSTM and HybridWeatherModel checkpoints: the model type
is read from the 'model_type' key stored by the updated train.py, with a
try/except fallback for older checkpoints that lack this key.

Usage (from project root):
    python src/evaluate.py
    python src/evaluate.py --model-path models/best_model.pth
"""

import sys
import os

# Ensure sibling modules (models/) are importable whether the script is
# invoked from the project root or from inside src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import pickle

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in all envs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.lstm_model import WeatherLSTM
from models.hybrid_model import HybridWeatherModel


# ============================================================================
# ModelEvaluator
# ============================================================================

class ModelEvaluator:
    """
    End-to-end evaluator for a trained temperature forecasting model.

    Workflow:
      1. Load checkpoint → detect model type → reconstruct model.
      2. Load test split (.npy arrays).
      3. Run batch inference → denormalise predictions and targets.
      4. Compute MAE, RMSE, R², MAPE.
      5. Save four diagnostic plots, a training-history plot, a JSON
         metrics file and a human-readable text report.
    """

    def __init__(self, model_path: str, data_dir: str = "data/processed"):
        """
        Initialise evaluator and load the model from checkpoint.

        Reads the 'model_type' key written by the updated train.py to
        decide which class to instantiate.  Falls back to a try/except
        heuristic for checkpoints produced by the original train.py.

        Args:
            model_path: Path to the .pth checkpoint file.
            data_dir:   Directory containing scaler.pkl, metadata.pkl and
                        the preprocessed .npy arrays.
        """
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir = data_dir

        # ── Load preprocessing artefacts ──────────────────────────────────
        with open(f"{data_dir}/metadata.pkl", "rb") as fh:
            self.metadata = pickle.load(fh)
        with open(f"{data_dir}/scaler.pkl", "rb") as fh:
            self.scaler = pickle.load(fh)

        n_features: int = int(self.metadata.get("n_features", 10))

        # ── Load checkpoint ───────────────────────────────────────────────
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_type = checkpoint.get("model_type", None)   # may be None for old checkpoints
        model_config: dict = checkpoint.get("model_config", {})

        self.model_type = model_type or "unknown"
        self.model      = self._build_model(model_type, model_config, n_features, state_dict)
        self.model.to(self.device).eval()

        # Declared here so Pylance knows these exist; populated by subsequent methods
        self.X_test_tensor: torch.Tensor
        self.y_test_norm:   np.ndarray
        self.y_pred_norm:   np.ndarray
        self.y_pred:        np.ndarray
        self.y_true:        np.ndarray
        self.residuals:     np.ndarray
        self.metrics:       dict

        print(f"[+] Model '{self.model_type}' loaded  [{self.device}]")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(
        self,
        model_type: str | None,
        model_config: dict,
        n_features: int,
        state_dict: dict,
    ) -> torch.nn.Module:
        """
        Reconstruct the model from checkpoint metadata.

        Priority:
          1. Use model_type + model_config from the checkpoint (new train.py).
          2. Try HybridWeatherModel (may fail if keys don't match).
          3. Fall back to WeatherLSTM.

        Args:
            model_type:   'lstm', 'hybrid' or None.
            model_config: Constructor kwargs dict (may be empty for old checkpoints).
            n_features:   Number of input features.
            state_dict:   Weights to load.

        Returns:
            Instantiated nn.Module with weights loaded.
        """
        # ── Explicit type from checkpoint (new format) ───────────────────
        if model_type == "hybrid":
            cfg = model_config or dict(
                input_size=n_features, lstm_hidden=128, lstm_layers=2,
                tcn_channels=64, tcn_levels=4, transformer_d_model=64,
                transformer_heads=4, transformer_layers=2, dropout=0.2,
            )
            # Convert numeric params to int, preserve dropout as float
            cfg_typed = {}
            for k, v in cfg.items():
                if k == "dropout":
                    cfg_typed[k] = float(v)
                else:
                    cfg_typed[k] = int(v)
            model = HybridWeatherModel(**cfg_typed)
            model.load_state_dict(state_dict)
            self.model_type = "hybrid"
            return model

        if model_type == "lstm":
            cfg = model_config or dict(
                input_size=n_features, hidden_size=128, num_layers=2, dropout=0.2
            )
            # Convert numeric params to int, preserve dropout as float
            cfg_typed = {}
            for k, v in cfg.items():
                if k == "dropout":
                    cfg_typed[k] = float(v)
                else:
                    cfg_typed[k] = int(v)
            model = WeatherLSTM(**cfg_typed)
            model.load_state_dict(state_dict)
            self.model_type = "lstm"
            return model

        # ── Fallback: try Hybrid first ────────────────────────────────────
        try:
            model = HybridWeatherModel(input_size=n_features)
            model.load_state_dict(state_dict)
            self.model_type = "hybrid (auto-detected)"
            return model
        except Exception:
            pass

        model = WeatherLSTM(input_size=n_features, hidden_size=128, num_layers=2, dropout=0.2)
        model.load_state_dict(state_dict)
        self.model_type = "lstm (auto-detected)"
        return model

    def _denorm_temperature(self, norm_values: np.ndarray) -> np.ndarray:
        """
        Inverse-transform a 1-D array of normalised temperature values to °C.

        The MinMaxScaler was fitted on all 10 features at once, so we embed
        the temperature column into a zero-filled dummy matrix and call
        inverse_transform; the result for the temperature column is correct
        regardless of the other columns being zero.

        Args:
            norm_values: 1-D numpy array of normalised temperature values.

        Returns:
            1-D numpy array of temperatures in °C.
        """
        n = len(norm_values)
        feature_names = list(self.scaler.feature_names_in_)
        temp_idx = feature_names.index("temperature")
        n_feat   = len(feature_names)

        dummy = np.zeros((n, n_feat), dtype=np.float32)
        dummy[:, temp_idx] = norm_values
        return self.scaler.inverse_transform(dummy)[:, temp_idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_test_data(self) -> None:
        """Load the held-out test split from disk."""
        X_test = np.load(f"{self.data_dir}/X_test.npy")
        y_test = np.load(f"{self.data_dir}/y_test.npy")

        self.X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        self.y_test_norm   = y_test          # normalised ground truth
        print(f"[+] Test: {len(y_test):,} samples")

    def run_inference(self) -> None:
        """
        Run batch inference on the test set and denormalise both predictions
        and ground-truth targets back to degrees Celsius.

        Populates:
            self.y_pred_norm  – raw normalised model output  (1-D ndarray)
            self.y_pred       – denormalised predictions  °C (1-D ndarray)
            self.y_true       – denormalised ground truth °C (1-D ndarray)
            self.residuals    – y_true − y_pred           °C (1-D ndarray)
        """
        print("[*] Running inference...")
        with torch.no_grad():
            self.y_pred_norm = self.model(self.X_test_tensor).cpu().numpy().flatten()

        self.y_pred    = self._denorm_temperature(self.y_pred_norm)
        self.y_true    = self._denorm_temperature(self.y_test_norm)
        self.residuals = self.y_true - self.y_pred
        print("[+] Done")

    def calculate_metrics(self) -> dict:
        """
        Compute MAE, RMSE, R² and MAPE on the denormalised test set.

        MAPE is computed only on samples where |y_true| > 0.5°C to avoid
        division by near-zero values that produce inf/nan.

        Returns:
            Dict with keys: mae, rmse, r2, mape, mae_norm, rmse_norm, r2_norm.
        """
        mae  = mean_absolute_error(self.y_true, self.y_pred)
        rmse = float(np.sqrt(mean_squared_error(self.y_true, self.y_pred)))
        r2   = r2_score(self.y_true, self.y_pred)

        # Safe MAPE: exclude near-zero temperatures (|T| ≤ 0.5°C)
        mask = np.abs(self.y_true) > 0.5
        if mask.sum() > 0:
            mape = float(
                np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask]))
                * 100
            )
        else:
            mape = float("nan")

        mae_n  = mean_absolute_error(self.y_test_norm, self.y_pred_norm)
        rmse_n = float(np.sqrt(mean_squared_error(self.y_test_norm, self.y_pred_norm)))
        r2_n   = r2_score(self.y_test_norm, self.y_pred_norm)

        self.metrics = {
            "mae":       mae,
            "rmse":      rmse,
            "r2":        r2,
            "mape":      mape,
            "mae_norm":  mae_n,
            "rmse_norm": rmse_n,
            "r2_norm":   r2_n,
        }

        print(f"\n[+] Metrics [{self.model_type}]:")
        print(f"   MAE  : {mae:.4f} C")
        print(f"   RMSE : {rmse:.4f} C")
        print(f"   R2   : {r2:.4f}")
        mape_str = f"{mape:.2f}%" if not np.isnan(mape) else "n/a (near-zero temps)"
        print(f"   MAPE : {mape_str}")

        return self.metrics

    def save_metrics_json(self, save_path: str = "docs/metrics.json") -> None:
        """
        Persist the computed metrics to JSON so that app.py can load real
        numbers instead of using hard-coded fallback values.

        Args:
            save_path: Destination path for the JSON file.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        payload = {**self.metrics, "model_type": self.model_type}
        with open(save_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[+] Metrics saved: {save_path}")

    def plot_predictions(
        self, save_path: str = "docs/predictions.png", n_samples: int = 500
    ) -> None:
        """
        Save a 2×2 diagnostic figure:
          1. Time-series: actual vs predicted (first n_samples test points).
          2. Scatter: predicted vs actual with ideal diagonal.
          3. Residual histogram with zero-error marker.
          4. Residuals vs predictions scatter.

        Args:
            save_path: Output path for the PNG.
            n_samples: Number of test points to plot in the time-series panel.
        """
        n = min(n_samples, len(self.y_true))
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1 — Time series
        ax = axes[0, 0]
        ax.plot(self.y_true[:n],        label="Реальные",       linewidth=1.5, alpha=0.8)
        ax.plot(self.y_pred[:n],        label="Предсказания",   linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Индекс времени")
        ax.set_ylabel("Температура (°C)")
        ax.set_title("Предсказания vs Реальные значения", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2 — Scatter
        ax = axes[0, 1]
        ax.scatter(self.y_true, self.y_pred, alpha=0.4, s=8, color="#1f77b4")
        lims = [self.y_true.min(), self.y_true.max()]
        ax.plot(lims, lims, "r--", linewidth=2, label="Идеальная линия")
        ax.set_xlabel("Реальная температура (°C)")
        ax.set_ylabel("Предсказанная температура (°C)")
        ax.set_title(f"Scatter  (R² = {self.metrics['r2']:.4f})", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3 — Residual histogram
        ax = axes[1, 0]
        ax.hist(self.residuals, bins=60, edgecolor="black", alpha=0.75, color="#1f77b4")
        ax.axvline(0, color="red",   linestyle="--", linewidth=2, label="Нулевая ошибка")
        ax.axvline(self.residuals.mean(), color="green", linestyle=":", linewidth=2,
                   label=f"Bias = {self.residuals.mean():.3f}°C")
        ax.set_xlabel("Ошибка предсказания (°C)")
        ax.set_ylabel("Частота")
        ax.set_title(f"Распределение ошибок  (MAE = {self.metrics['mae']:.2f}°C)",
                     fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4 — Residuals vs predictions
        ax = axes[1, 1]
        ax.scatter(self.y_pred, self.residuals, alpha=0.4, s=8, color="#ff7f0e")
        ax.axhline(0,                   color="red",    linestyle="--", linewidth=2)
        ax.axhline( self.metrics["mae"], color="orange", linestyle=":",  linewidth=1.5,
                    label=f"+MAE = {self.metrics['mae']:.2f}°C")
        ax.axhline(-self.metrics["mae"], color="orange", linestyle=":",  linewidth=1.5,
                    label=f"−MAE = {self.metrics['mae']:.2f}°C")
        ax.set_xlabel("Предсказанная температура (°C)")
        ax.set_ylabel("Ошибка (°C)")
        ax.set_title("Остатки vs Предсказания", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Оценка модели: {self.model_type}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] Prediction plots saved: {save_path}")

    def plot_training_history(self, save_path: str = "docs/training_history.png") -> None:
        """
        Plot loss curves and learning-rate schedule from training_history.json.

        Skips gracefully if the history file is not found.

        Args:
            save_path: Output path for the PNG.
        """
        history_path = "models/training_history.json"
        if not os.path.exists(history_path):
            print("[!] Training history not found, skipping.")
            return

        with open(history_path, "r") as fh:
            history = json.load(fh)

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
        ax1.plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.set_title("Кривые обучения", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, history["learning_rate"], linewidth=2, color="green")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule", fontweight="bold")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"История обучения: {self.model_type}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[+] Training history plot saved: {save_path}")

    def generate_report(self, save_path: str = "docs/evaluation_report.txt") -> None:
        """
        Write a structured plain-text evaluation report for the thesis appendix.

        Args:
            save_path: Destination text file path.
        """
        mape_val = self.metrics["mape"]
        if np.isnan(mape_val):
            mape_str = "n/a (значения близки к нулю)"
        else:
            mape_note = " (некорректна для рядов с T≈0°C)" if mape_val > 20 else ""
            mape_str = f"{mape_val:.2f}%{mape_note}"
        quality = (
            "Отличная" if self.metrics["r2"] > 0.90
            else "Хорошая" if self.metrics["r2"] > 0.80
            else "Удовлетворительная"
        )
        verdict = "успешно" if self.metrics["mae"] < 3.0 else "удовлетворительно"

        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║       ОТЧЁТ ОБ ОЦЕНКЕ МОДЕЛИ ПРОГНОЗИРОВАНИЯ ПОГОДЫ              ║
╚══════════════════════════════════════════════════════════════════╝

📅 Дата оценки  : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
🧠 Архитектура  : {self.model_type}

📊 ДАННЫЕ:
   • Тестовых примеров    : {len(self.y_true):,}
   • Признаков            : {self.metadata['n_features']}
   • Длина последоват.    : {self.metadata['sequence_length']} ч

📈 МЕТРИКИ (реальные °C):
   • MAE   : {self.metrics['mae']:.4f} °C
   • RMSE  : {self.metrics['rmse']:.4f} °C
   • R²    : {self.metrics['r2']:.4f}
   • MAPE  : {mape_str}

📈 МЕТРИКИ (нормализованные):
   • MAE   : {self.metrics['mae_norm']:.6f}
   • RMSE  : {self.metrics['rmse_norm']:.6f}
   • R²    : {self.metrics['r2_norm']:.4f}

💡 ИНТЕРПРЕТАЦИЯ:
   • R² = {self.metrics['r2']:.4f} → {quality} корреляция
     ({self.metrics['r2'] * 100:.1f}% дисперсии объяснено моделью)
   • MAE = {self.metrics['mae']:.4f}°C → средняя абсолютная ошибка
   • Bias = {self.residuals.mean():.4f}°C → систематическое смещение

✅ ЗАКЛЮЧЕНИЕ:
   Модель {verdict} прогнозирует температуру в Астане.
   Средняя абсолютная ошибка составляет {self.metrics['mae']:.2f}°C
   на горизонте прогнозирования {self.metadata['sequence_length']} часов.
"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"[+] Report saved: {save_path}")


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained weather forecasting model."
    )
    parser.add_argument(
        "--model-path", default="models/best_model.pth", dest="model_path",
        help="Path to the .pth checkpoint (default: models/best_model.pth)",
    )
    parser.add_argument(
        "--data-dir", default="data/processed", dest="data_dir",
        help="Directory with preprocessed data (default: data/processed)",
    )
    args = parser.parse_args()

    print("[*] EVALUATION — Weather Forecasting Model\n")

    ev = ModelEvaluator(model_path=args.model_path, data_dir=args.data_dir)
    ev.load_test_data()
    ev.run_inference()
    ev.calculate_metrics()
    ev.save_metrics_json("docs/metrics.json")
    ev.plot_predictions("docs/predictions.png")
    ev.plot_training_history("docs/training_history.png")
    ev.generate_report("docs/evaluation_report.txt")

    print("\n[+] Evaluation complete!")
