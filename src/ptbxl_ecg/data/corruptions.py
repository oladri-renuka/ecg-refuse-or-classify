"""Test-time corruptions in raw millivolt space (C6 robustness, streaming demo)."""
from __future__ import annotations

import numpy as np


def gaussian_noise(x: np.ndarray, sigma_mv: float, rng: np.random.Generator) -> np.ndarray:
    return x + rng.normal(0.0, sigma_mv, size=x.shape).astype(np.float32) if sigma_mv > 0 else x


def baseline_wander(x: np.ndarray, amp_mv: float, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Sum of two low-frequency sines (0.05-0.5 Hz) per record, lead-specific gain."""
    if amp_mv <= 0:
        return x
    *batch, C, T = x.shape
    n = int(np.prod(batch)) if batch else 1
    t = np.arange(T) / fs
    wander = np.zeros((n, C, T), dtype=np.float32)
    for _ in range(2):
        f = rng.uniform(0.05, 0.5, size=(n, 1, 1))
        phase = rng.uniform(0, 2 * np.pi, size=(n, 1, 1))
        gain = rng.uniform(0.5, 1.0, size=(n, C, 1))
        wander += (0.5 * amp_mv * gain * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)
    return x + wander.reshape(x.shape)


def lead_loss(x: np.ndarray, n_leads: int, rng: np.random.Generator) -> np.ndarray:
    """Flat-line `n_leads` random leads per record (disconnected electrodes)."""
    if n_leads <= 0:
        return x
    x = x.copy()
    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    for rec in flat:
        rec[rng.choice(rec.shape[0], size=n_leads, replace=False)] = 0.0
    return flat.reshape(x.shape)


def apply(kind: str, x: np.ndarray, level: float, fs: float, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian_mv":
        return gaussian_noise(x, level, rng)
    if kind == "baseline_wander_mv":
        return baseline_wander(x, level, fs, rng)
    if kind == "lead_loss":
        return lead_loss(x, int(level), rng)
    raise ValueError(kind)
