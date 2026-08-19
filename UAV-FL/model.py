"""
model.py — Lightweight Hybrid CNN-LSTM Anomaly Detector (paper Section III-C).

Architecture (matches Fig. 1 of the paper):
    Input X in R^(w x d)
      -> Conv1D(32, k=3, ReLU) -> BatchNorm
      -> Conv1D(64, k=3, ReLU) -> BatchNorm -> MaxPool(2)
      -> LSTM(64, return_sequences=True) -> Dropout(0.3)
      -> LSTM(32, return_sequences=False) -> Dropout(0.3)
      -> Dense(16, ReLU)
      -> Dense(1, Sigmoid)               -> anomaly probability

This is the "full" production architecture referenced throughout the paper
as Section III-C. It is intentionally small (targeting the ~180 KB
post-compaction budget discussed in the paper) so that it is realistic to
deploy on a Jetson-class or Raspberry-Pi-class UAV companion computer.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CNNLSTMAnomalyDetector(nn.Module):
    def __init__(self, n_features: int = 28, conv1_filters: int = 32, conv2_filters: int = 64,
                 lstm1_units: int = 64, lstm2_units: int = 32, dense_units: int = 16,
                 dropout: float = 0.3):
        super().__init__()
        self.n_features = n_features

        # ---- CNN feature extraction (operates over the time axis) ----
        self.conv1 = nn.Conv1d(n_features, conv1_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.conv2 = nn.Conv1d(conv1_filters, conv2_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.relu = nn.ReLU()

        # ---- LSTM temporal sequence modeling ----
        self.lstm1 = nn.LSTM(input_size=conv2_filters, hidden_size=lstm1_units, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=lstm1_units, hidden_size=lstm2_units, batch_first=True)
        self.drop2 = nn.Dropout(dropout)

        # ---- Classification head ----
        self.fc1 = nn.Linear(lstm2_units, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, window, n_features) -> (batch,) anomaly probability."""
        # Conv1d expects (batch, channels=features, length=time)
        z = x.transpose(1, 2)                      # (B, d, w)
        z = self.relu(self.bn1(self.conv1(z)))      # (B, 32, w)
        z = self.relu(self.bn2(self.conv2(z)))      # (B, 64, w)
        z = self.pool(z)                            # (B, 64, w/2)
        z = z.transpose(1, 2)                        # (B, w/2, 64) for LSTM

        z, _ = self.lstm1(z)                         # (B, w/2, 64)
        z = self.drop1(z)
        z, (h_n, _) = self.lstm2(z)                   # h_n: (1, B, 32)
        z = self.drop2(h_n.squeeze(0))                # (B, 32)

        z = self.relu(self.fc1(z))                    # (B, 16)
        logit = self.fc2(z).squeeze(-1)                # (B,)
        return torch.sigmoid(logit)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def quantized_size_bytes(model: nn.Module, bits: int = 8) -> int:
    """Estimated on-disk size after post-training quantization to `bits`."""
    n_params = count_parameters(model)
    return int(n_params * bits / 8)


if __name__ == "__main__":
    m = CNNLSTMAnomalyDetector(n_features=28)
    n = count_parameters(m)
    print(f"Parameters: {n:,}  |  FP32 size: {n * 4 / 1024:.1f} KB  |  "
          f"INT8 size: {quantized_size_bytes(m):,} bytes ({quantized_size_bytes(m)/1024:.1f} KB)")
    x = torch.randn(8, 50, 28)
    y = m(x)
    print("Forward output shape:", y.shape, " sample:", y[:3].detach().numpy())
