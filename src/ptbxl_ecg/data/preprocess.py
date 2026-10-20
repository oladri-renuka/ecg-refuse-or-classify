"""Signal preprocessing shared by training, evaluation and the serving pipeline."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_sos(fs: float, low: float = 0.5, high: float = 40.0, order: int = 3) -> np.ndarray:
    return butter(order, [low, high], btype="band", fs=fs, output="sos")


def bandpass(x: np.ndarray, fs: float, chunk: int = 2048) -> np.ndarray:
    """Zero-phase 0.5-40 Hz filter along the last axis; chunked to bound memory."""
    sos = bandpass_sos(fs)
    if x.ndim < 3:
        return sosfiltfilt(sos, x, axis=-1).astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    for i in range(0, len(x), chunk):
        out[i : i + chunk] = sosfiltfilt(sos, x[i : i + chunk], axis=-1)
    return out


def compute_norm_stats(X: np.ndarray, idx: np.ndarray, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead mean/std over the given (training) records only."""
    n_leads = X.shape[1]
    total = np.zeros(n_leads)
    total_sq = np.zeros(n_leads)
    count = 0
    for i in range(0, len(idx), chunk):
        block = X[idx[i : i + chunk]].astype(np.float64)
        total += block.sum(axis=(0, 2))
        total_sq += (block**2).sum(axis=(0, 2))
        count += block.shape[0] * block.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[:, None]) / std[:, None]).astype(np.float32)


def preprocess(x_raw: np.ndarray, mean: np.ndarray, std: np.ndarray, fs: float, use_bandpass: bool) -> np.ndarray:
    x = bandpass(x_raw, fs) if use_bandpass else x_raw
    return standardize(x, mean, std)
