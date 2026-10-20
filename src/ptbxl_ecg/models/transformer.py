"""Patch Transformer for ECG. Attention is written out explicitly (no fused
kernels) so ONNX export and CPU inference behave identically everywhere."""
from __future__ import annotations

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        attn = self.drop(torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, D))


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_ratio * dim, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pre-norm
        x = x + self.drop(self.attn(self.norm1(x)))
        return x + self.drop(self.mlp(self.norm2(x)))


class ECGTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        n_classes: int = 5,
        patch: int = 25,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_channels, d_model, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_len // patch + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(EncoderLayer(d_model, heads, mlp_ratio, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch_embed(x).transpose(1, 2)            # (B, N, D)
        z = torch.cat([self.cls.expand(z.shape[0], -1, -1), z], dim=1)
        z = z + self.pos[:, : z.shape[1]]
        for layer in self.layers:
            z = layer(z)
        return self.head(self.norm(z[:, 0]))
