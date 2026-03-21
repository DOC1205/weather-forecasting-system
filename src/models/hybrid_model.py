"""
Hybrid LSTM + TCN + Transformer model for weather forecasting in Astana.

Architecture overview:
  - LSTM branch   : captures long-range temporal dependencies
  - TCN branch    : captures local patterns via dilated causal convolutions
  - Transformer   : captures global attention over the 24-hour window
  - Gated fusion  : learned soft-attention weights combine the three branches
  - Shared head   : FC layers produce the final temperature prediction

Reference: diploma thesis "Weather Forecasting System for Astana using Hybrid Deep Learning"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as introduced in 'Attention Is All You Need'.

    Adds a fixed, non-learnable position signal to the token embeddings so that
    the Transformer can distinguish different time-steps within the sequence.

    Args:
        d_model:  Embedding dimension (must match TransformerEncoderLayer d_model).
        dropout:  Dropout applied after adding positional encoding.
        max_len:  Maximum sequence length the encoder can handle.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added, shape unchanged.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# TCN: Temporal Convolutional Network
# ---------------------------------------------------------------------------

class _CausalBlock(nn.Module):
    """
    Single TCN residual block using dilated causal convolutions.

    Causal padding ensures the model never looks into the future:
    we pad (kernel_size - 1) * dilation on the left only, then trim
    the right to restore the original sequence length.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels.
        kernel_size:  Convolution kernel size.
        dilation:     Dilation factor (doubles each level: 1, 2, 4, 8 …).
        dropout:      Dropout probability applied after each activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 1×1 projection for the residual shortcut when channel dims differ
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, seq_len]
        Returns:
            out: [batch, out_channels, seq_len]
        """
        # Causal convolution: trim extra future padding
        out = self.relu(self.conv1(x)[:, :, : x.size(2)])
        out = self.dropout(out)
        out = self.relu(self.conv2(out)[:, :, : x.size(2)])
        out = self.dropout(out)

        res = x if self.residual_proj is None else self.residual_proj(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder.

    Stacks exponentially dilated causal blocks to achieve a large effective
    receptive field without increasing depth linearly.

    Args:
        input_size:   Number of input features (channels).
        num_channels: Internal channel width for all TCN blocks.
        kernel_size:  Convolution kernel size (default 3).
        num_levels:   Number of dilated blocks; receptive field = 2^num_levels.
        dropout:      Dropout for regularisation.

    Output:
        Last time-step representation of shape [batch, num_channels].
    """

    def __init__(
        self,
        input_size: int,
        num_channels: int = 64,
        kernel_size: int = 3,
        num_levels: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels
            layers.append(
                _CausalBlock(in_ch, num_channels, kernel_size, dilation, dropout)
            )
        self.network = nn.Sequential(*layers)
        self.output_size = num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]  (batch-first convention)
        Returns:
            last: [batch, num_channels]
        """
        x = x.transpose(1, 2)          # → [batch, features, seq_len]
        out = self.network(x)           # → [batch, num_channels, seq_len]
        return out[:, :, -1]            # last time-step


# ---------------------------------------------------------------------------
# Transformer Encoder branch
# ---------------------------------------------------------------------------

class TransformerEncoderBranch(nn.Module):
    """
    Lightweight Transformer encoder for temporal weather sequences.

    Projects the raw feature vector into a d_model-dimensional space, adds
    sinusoidal positional encoding, then runs multi-head self-attention over
    the 24-hour window.

    Args:
        input_size:       Number of raw input features (11).
        d_model:          Internal embedding dimension (must be divisible by nhead).
        nhead:            Number of attention heads.
        num_layers:       Number of stacked TransformerEncoderLayer blocks.
        dropout:          Dropout in attention and feed-forward sub-layers.

    Output:
        Last time-step representation of shape [batch, d_model].
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_size = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, features]
        Returns:
            [batch, d_model]
        """
        x = self.input_proj(x)     # [batch, seq_len, d_model]
        x = self.pos_encoding(x)
        out = self.transformer(x)  # [batch, seq_len, d_model]
        return out[:, -1, :]       # last time-step: [batch, d_model]


# ---------------------------------------------------------------------------
# Hybrid Model
# ---------------------------------------------------------------------------

class HybridWeatherModel(nn.Module):
    """
    Hybrid LSTM + TCN + Transformer model for next-hour temperature forecasting.

    Each of the three branches independently encodes the 24-hour input sequence.
    A learned gating network computes soft attention weights over the branches,
    and the weighted sum of projected representations feeds the shared
    prediction head.

    Architecture summary:
        Input [B, 24, 11]
            ├── LSTM branch   → [B, 128]
            ├── TCN branch    → [B,  64]
            └── Transformer   → [B,  64]
                      │
                 Gated fusion (softmax over 3 gates)
                      │
               Shared head  FC 128→64→32→1
                      │
               Temperature prediction  [B, 1]

    Args:
        input_size:          Number of input features (default 11).
        lstm_hidden:         LSTM hidden state dimension.
        lstm_layers:         Number of stacked LSTM layers.
        tcn_channels:        Channel width for TCN blocks.
        tcn_levels:          Number of TCN dilated levels (receptive field = 2^levels).
        transformer_d_model: Transformer embedding dimension.
        transformer_heads:   Number of attention heads.
        transformer_layers:  Number of Transformer encoder layers.
        dropout:             Shared dropout rate across all branches.
    """

    def __init__(
        self,
        input_size: int = 11,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        tcn_channels: int = 64,
        tcn_levels: int = 4,
        transformer_d_model: int = 64,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ---- LSTM branch ------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        lstm_out = lstm_hidden

        # ---- TCN branch -------------------------------------------------
        self.tcn = TCNEncoder(
            input_size=input_size,
            num_channels=tcn_channels,
            kernel_size=3,
            num_levels=tcn_levels,
            dropout=dropout,
        )
        tcn_out = tcn_channels

        # ---- Transformer branch -----------------------------------------
        self.transformer = TransformerEncoderBranch(
            input_size=input_size,
            d_model=transformer_d_model,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
        )
        trans_out = transformer_d_model

        # ---- Gated fusion -----------------------------------------------
        fusion_dim = 128
        total_concat = lstm_out + tcn_out + trans_out

        # Gate: maps concatenated branch outputs → 3 scalar weights (softmax)
        self.gate = nn.Sequential(
            nn.Linear(total_concat, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )

        # Project each branch to a common fusion dimension
        self.lstm_proj = nn.Linear(lstm_out, fusion_dim)
        self.tcn_proj  = nn.Linear(tcn_out,  fusion_dim)
        self.trans_proj = nn.Linear(trans_out, fusion_dim)

        # ---- Prediction head --------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, return_gates: bool = False):
        """
        Forward pass through all three branches, gated fusion, and prediction head.

        Args:
            x:            Input tensor of shape [batch_size, sequence_length, input_size].
                          sequence_length = 24 hours, input_size = 11 features.
            return_gates: If True, also return gate weights as a numpy array [batch, 3].

        Returns:
            Predicted temperature offset (normalised), shape [batch_size, 1].
            If return_gates=True, returns (prediction, gates_np) tuple.
        """
        # --- LSTM: take last hidden state of the top layer ---------------
        _, (h_n, _) = self.lstm(x)
        lstm_feat = h_n[-1]              # [B, lstm_hidden]

        # --- TCN: last time-step output -----------------------------------
        tcn_feat = self.tcn(x)           # [B, tcn_channels]

        # --- Transformer: last token representation ----------------------
        trans_feat = self.transformer(x) # [B, d_model]

        # --- Gated fusion ------------------------------------------------
        concat = torch.cat([lstm_feat, tcn_feat, trans_feat], dim=-1)  # [B, total]
        gates = self.gate(concat)        # [B, 3]

        fused = (
            gates[:, 0:1] * self.lstm_proj(lstm_feat)
            + gates[:, 1:2] * self.tcn_proj(tcn_feat)
            + gates[:, 2:3] * self.trans_proj(trans_feat)
        )                                # [B, fusion_dim]

        prediction = self.head(fused)    # [B, 1]

        if return_gates:
            return prediction, gates.detach().cpu().numpy()
        return prediction

    def get_branch_weights(self, x: torch.Tensor) -> dict:
        """
        Utility: return the average gating weights for interpretability.

        Useful in the thesis to show how much each branch contributes
        on a given batch.

        Args:
            x: [batch, seq_len, features]
        Returns:
            dict with keys 'lstm', 'tcn', 'transformer' and float values
            summing to 1.0.
        """
        self.eval()
        with torch.no_grad():
            _, (h_n, _) = self.lstm(x)
            lstm_feat  = h_n[-1]
            tcn_feat   = self.tcn(x)
            trans_feat = self.transformer(x)
            concat = torch.cat([lstm_feat, tcn_feat, trans_feat], dim=-1)
            gates = self.gate(concat).mean(dim=0)   # average over batch
        return {
            "lstm":        gates[0].item(),
            "tcn":         gates[1].item(),
            "transformer": gates[2].item(),
        }


# ---------------------------------------------------------------------------
# Parameter count helper
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🧪 Testing HybridWeatherModel …\n")

    B, T, F = 8, 24, 11
    x = torch.randn(B, T, F)

    model = HybridWeatherModel(input_size=F)
    model.eval()

    with torch.no_grad():
        out = model(x)

    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Params : {count_parameters(model):,}")
    print(f"\nBranch weights (sample): {model.get_branch_weights(x)}")
    print("\n✅ HybridWeatherModel OK")
