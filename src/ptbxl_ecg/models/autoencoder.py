"""Conv autoencoder for the signal-quality gate: 12x1000 -> 128x125 -> 12x1000."""
from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int, k: int, stride: int = 1) -> list[nn.Module]:
    return [nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False), nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 12, channels: tuple[int, ...] = (32, 64, 128), kernel: int = 7):
        super().__init__()
        enc, c_prev = [], in_channels
        for c in channels:  # each stage halves the length
            enc += _block(c_prev, c, kernel, stride=2) + _block(c, c, kernel)
            c_prev = c
        self.encoder = nn.Sequential(*enc)

        dec, ups = [], list(channels[::-1][1:]) + [channels[0]]
        for c in ups:  # mirrored: transposed conv doubles the length
            dec += [nn.ConvTranspose1d(c_prev, c, 4, stride=2, padding=1, bias=False), nn.BatchNorm1d(c), nn.ReLU(inplace=True)]
            dec += _block(c, c, kernel)
            c_prev = c
        dec.append(nn.Conv1d(c_prev, in_channels, kernel, padding=kernel // 2))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Per-record mean squared error, shape (B,)."""
    return ((x - x_hat) ** 2).mean(dim=(1, 2))


class AEScorer(nn.Module):
    """Export wrapper: standardized ECG in, per-record reconstruction error out."""

    def __init__(self, ae: ConvAutoencoder):
        super().__init__()
        self.ae = ae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return reconstruction_error(x, self.ae(x))
