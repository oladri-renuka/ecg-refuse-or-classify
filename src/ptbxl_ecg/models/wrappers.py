from __future__ import annotations

import torch
from torch import nn


def window_starts(total_len: int, crop_len: int, stride: int) -> list[int]:
    if crop_len >= total_len:
        return [0]
    starts = list(range(0, total_len - crop_len + 1, stride))
    if starts[-1] != total_len - crop_len:
        starts.append(total_len - crop_len)  # always cover the tail
    return starts


class SlidingWindowClassifier(nn.Module):
    """Full record in, window-averaged sigmoid probabilities out.

    Model is trained on random crops; at test time we score fixed overlapping
    windows and average, as in the PTB-XL benchmark. This is also the exported
    ONNX graph, so latency numbers include all windows."""

    def __init__(self, model: nn.Module, crop_len: int, stride: int, total_len: int = 1000):
        super().__init__()
        self.model = model
        self.crop_len = crop_len
        self.starts = window_starts(total_len, crop_len, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        windows = torch.cat([x[:, :, s : s + self.crop_len] for s in self.starts], dim=0)
        probs = torch.sigmoid(self.model(windows))
        return probs.view(len(self.starts), -1, probs.shape[-1]).mean(0)
