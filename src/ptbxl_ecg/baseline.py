"""B0 honest baseline: handcrafted ECG features + one gradient-boosted tree model per class.

Features (per record, from 0.5-40 Hz filtered raw signal):
  * per-lead amplitude statistics (percentiles, std, skew, kurtosis, mean |diff|)
  * per-lead log band powers (Welch)
  * rhythm: beat count, mean/std RR, RMSSD from a multi-lead R-peak detector
  * per-lead median-beat template (-250 ms .. +450 ms around R, decimated x2),
    which carries ST elevation/depression, T inversion, QRS width and voltage

Launched like every other experiment: python -m ptbxl_ecg.train configs/baseline.yaml --seeds 0 1 2
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, skew

from ptbxl_ecg.data.dataset import load_raw
from ptbxl_ecg.data.preprocess import bandpass
from ptbxl_ecg.metrics import classification_report, macro_auroc

FEATURE_VERSION = 1
BANDS = [(0.5, 4), (4, 8), (8, 15), (15, 25), (25, 40)]
PCTS = [1, 5, 25, 50, 75, 95, 99]


def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """x: (12, T) filtered. Peaks of the summed absolute slope across leads."""
    energy = np.abs(np.diff(x, axis=1)).sum(0)
    if energy.max() <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(energy, distance=int(0.3 * fs), height=0.35 * np.percentile(energy, 99))
    return peaks


def record_features(x: np.ndarray, fs: float) -> np.ndarray:
    pre, post = int(0.25 * fs), int(0.45 * fs)
    T = x.shape[1]
    stats = [
        np.percentile(x, PCTS, axis=1).T,                        # (12, 7)
        x.std(1, keepdims=True),
        skew(x, axis=1)[:, None],
        kurtosis(x, axis=1)[:, None],
        np.abs(np.diff(x, axis=1)).mean(1, keepdims=True),
    ]
    f, pxx = welch(x, fs=fs, nperseg=min(256, T), axis=1)
    bands = [np.log(pxx[:, (f >= lo) & (f < hi)].sum(1, keepdims=True) + 1e-10) for lo, hi in BANDS]

    peaks = detect_r_peaks(x, fs)
    peaks_ok = peaks[(peaks - pre >= 0) & (peaks + post < T)]
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        rhythm = [len(peaks), rr.mean(), rr.std(), np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else 0.0]
    else:
        rhythm = [len(peaks), 0.0, 0.0, 0.0]
    if len(peaks_ok):
        beats = np.stack([x[:, p - pre : p + post] for p in peaks_ok])  # (n_beats, 12, L)
        template = np.median(beats, axis=0)[:, ::2]
    else:
        template = np.zeros((x.shape[0], (pre + post + 1) // 2))
    return np.concatenate([np.hstack(stats + bands).ravel(), np.asarray(rhythm, dtype=float), template.ravel()]).astype(np.float32)


def extract_features(X_raw: np.ndarray, fs: float, cache: Path | None = None) -> np.ndarray:
    if cache is not None and cache.exists():
        feats = np.load(cache)
        if len(feats) == len(X_raw):
            return feats
    Xf = bandpass(X_raw, fs)
    feats = np.stack([record_features(x, fs) for x in Xf])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    if cache is not None:
        np.save(cache, feats)
    return feats


def _make_model(bcfg: dict, seed: int):
    if bcfg["model"] == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=bcfg["max_iter"], learning_rate=bcfg["learning_rate"], max_leaf_nodes=bcfg["max_leaf_nodes"],
            l2_regularization=bcfg["l2_regularization"], early_stopping=bcfg["early_stopping"], random_state=seed,
        )
    if bcfg["model"] == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(C=bcfg["C"], max_iter=3000, random_state=seed))
    raise ValueError(bcfg["model"])


def train_baseline(cfg: dict, out: Path) -> dict:
    X, Y, _, sp = load_raw(cfg)
    fs = cfg["data"]["sampling_rate"]
    cache = Path(cfg["data"]["cache_dir"]) / f"features_v{FEATURE_VERSION}_{fs}.npy"
    t0 = time.time()
    F = extract_features(X, fs, cache)
    del X

    p_val = np.zeros((len(sp.val), Y.shape[1]), dtype=np.float32)
    p_test = np.zeros((len(sp.test), Y.shape[1]), dtype=np.float32)
    for c in range(Y.shape[1]):
        clf = _make_model(cfg["baseline"], cfg["seed"]).fit(F[sp.train], Y[sp.train, c])
        p_val[:, c] = clf.predict_proba(F[sp.val])[:, 1]
        p_test[:, c] = clf.predict_proba(F[sp.test])[:, 1]
        print(f"[baseline] class {c} done", flush=True)

    y_val, y_test = Y[sp.val], Y[sp.test]
    np.savez_compressed(out / "preds.npz", val_idx=sp.val, test_idx=sp.test, p_val=p_val, p_test=p_test, y_val=y_val, y_test=y_test)
    return {
        "model": f"features+{cfg['baseline']['model']}",
        "n_features": int(F.shape[1]),
        "train_size": int(len(sp.train)),
        "train_time_s": time.time() - t0,
        "val": {"macro_auroc": macro_auroc(y_val, p_val)},
        "test": classification_report(y_test, p_test, y_val, p_val),
    }
