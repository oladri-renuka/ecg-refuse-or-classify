"""Batched, on-device crop + augmentation. Uses torch RNG so it is seeded by set_seed()."""
from __future__ import annotations

import math

import torch


def random_crop(x: torch.Tensor, length: int) -> torch.Tensor:
    """x: (B, C, T) -> (B, C, length), independent random start per sample."""
    B, C, T = x.shape
    if length >= T:
        return x
    starts = torch.randint(0, T - length + 1, (B,))
    idx = starts[:, None] + torch.arange(length)[None, :]
    return x.gather(2, idx[:, None, :].expand(B, C, length))


class BatchAugmenter:
    def __init__(self, acfg: dict | None, fs: float):
        self.cfg = acfg or {}
        self.fs = fs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        B, C, L = x.shape
        dev = x.device

        def uniform(lo, hi, *shape):
            return torch.empty(*shape, device=dev).uniform_(lo, hi)

        if c.get("amp_scale"):
            x = x * uniform(*c["amp_scale"], B, 1, 1)
        if c.get("noise_p", 0) > 0:
            on = (torch.rand(B, 1, 1, device=dev) < c["noise_p"]).float()
            sigma = uniform(*c["noise_sigma"], B, 1, 1)
            x = x + on * sigma * torch.randn_like(x)
        if c.get("wander_p", 0) > 0:
            on = (torch.rand(B, 1, 1, device=dev) < c["wander_p"]).float()
            freq = uniform(*c["wander_freq_hz"], B, 1, 1)
            phase = uniform(0.0, 2 * math.pi, B, 1, 1)
            amp = uniform(*c["wander_amp"], B, C, 1)
            t = torch.arange(L, device=dev).float()[None, None, :] / self.fs
            x = x + on * amp * torch.sin(2 * math.pi * freq * t + phase)
        if c.get("lead_dropout_p", 0) > 0:
            drop = torch.rand(B, C, 1, device=dev) < c["lead_dropout_p"]
            drop &= ~drop.all(dim=1, keepdim=True)  # never drop every lead
            x = x * (~drop).float()
        return x
