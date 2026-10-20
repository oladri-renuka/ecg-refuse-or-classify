from __future__ import annotations

import torch
from torch import nn


class BiLSTM(nn.Module):
    """Stacked bidirectional LSTM over time, mean + max pooling, linear head."""

    def __init__(self, in_channels: int = 12, n_classes: int = 5, hidden: int = 256, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            in_channels, hidden, num_layers=layers, batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        h, _ = self.lstm(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=1))
