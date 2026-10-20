"""xresnet1d-style 1D ResNet (Strodthoff et al. 2021 style) for 12-lead ECG."""
from __future__ import annotations

import torch
from torch import nn


def conv_bn(c_in: int, c_out: int, k: int, stride: int = 1, act: bool = True, zero_bn: bool = False) -> nn.Sequential:
    bn = nn.BatchNorm1d(c_out)
    if zero_bn:  # residual branch starts as identity (xresnet trick)
        nn.init.zeros_(bn.weight)
    layers = [nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False), bn]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, k: int):
        super().__init__()
        self.branch = nn.Sequential(conv_bn(c_in, c_out, k, stride), conv_bn(c_out, c_out, k, act=False, zero_bn=True))
        # Strided 1x1 projection. (xresnet's AvgPool(ceil_mode=True) shortcut is avoided:
        # ONNX pads the odd-length tail differently, which breaks export parity.)
        needs_proj = stride != 1 or c_in != c_out
        self.shortcut = conv_bn(c_in, c_out, 1, stride=stride, act=False) if needs_proj else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.branch(x) + self.shortcut(x))


class ResNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        n_classes: int = 5,
        widths: tuple[int, ...] = (64, 128, 256, 512),
        depths: tuple[int, ...] = (2, 2, 2, 2),
        kernel: int = 5,
        stem_width: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            conv_bn(in_channels, stem_width, kernel, stride=2),
            conv_bn(stem_width, stem_width, kernel),
            conv_bn(stem_width, widths[0], kernel),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        stages, c_prev = [], widths[0]
        for i, (width, depth) in enumerate(zip(widths, depths)):
            blocks = [BasicBlock(c_prev if j == 0 else width, width, 2 if (i > 0 and j == 0) else 1, kernel) for j in range(depth)]
            stages.append(nn.Sequential(*blocks))
            c_prev = width
        self.stages = nn.ModuleList(stages)
        self.head = nn.Sequential(nn.BatchNorm1d(2 * c_prev), nn.Dropout(dropout), nn.Linear(2 * c_prev, n_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))
