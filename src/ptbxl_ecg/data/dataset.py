"""Load the cache, split by strat_fold, fit standardization on train only."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ptbxl_ecg.data import DEFAULT_DATA, PIPELINE_VERSION
from ptbxl_ecg.data.preprocess import bandpass, compute_norm_stats


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class DataBundle:
    X: np.ndarray        # (N, 12, T) standardized float32
    Y: np.ndarray        # (N, 5)
    meta: pd.DataFrame   # row i <-> X[i]
    splits: Splits
    mean: np.ndarray     # (12,) fitted on splits.train
    std: np.ndarray
    fs: int


class PipelineVersionError(RuntimeError):
    pass


def check_cache_version(cache_dir: str | Path) -> None:
    info_path = Path(cache_dir) / "info.json"
    found = json.loads(info_path.read_text()).get("pipeline_version") if info_path.exists() else None
    if found != PIPELINE_VERSION:
        raise PipelineVersionError(
            f"{cache_dir} was built with pipeline_version={found}, this code is {PIPELINE_VERSION}. "
            "Rebuild it with `python -m ptbxl_ecg.data.ptbxl`."
        )


def load_cache(cache_dir: str | Path, fs: int = 100) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Raw cache: X (N, 12, T) in mV, Y (N, 5) multi-hot, meta (row i <-> X[i])."""
    cache_dir = Path(cache_dir)
    check_cache_version(cache_dir)
    X = np.load(cache_dir / f"X_{fs}.npy")
    Y = np.load(cache_dir / "Y.npy").astype(np.float32)
    meta = pd.read_csv(cache_dir / "meta.csv")
    assert len(X) == len(Y) == len(meta), "cache files are out of sync"
    return X, Y, meta


def make_splits(meta: pd.DataFrame, dcfg: dict | None = None) -> Splits:
    """Benchmark split by strat_fold. dcfg=None uses DEFAULT_DATA (folds 1-8 / 9 / 10)."""
    dcfg = {**DEFAULT_DATA, **(dcfg or {})}
    folds = meta["strat_fold"].to_numpy()
    train = np.flatnonzero(np.isin(folds, dcfg["train_folds"]))
    frac = float(dcfg.get("train_fraction", 1.0))
    if frac < 1.0:
        rng = np.random.default_rng(dcfg.get("subset_seed", 0))
        train = np.sort(rng.choice(train, size=max(1, round(frac * len(train))), replace=False))
    return Splits(
        train=train,
        val=np.flatnonzero(np.isin(folds, dcfg["val_folds"])),
        test=np.flatnonzero(np.isin(folds, dcfg["test_folds"])),
    )


def load_bundle(cfg: dict, stats: tuple[np.ndarray, np.ndarray] | None = None) -> DataBundle:
    """Pass `stats` to reuse a checkpoint's normalization instead of refitting."""
    dcfg = cfg["data"]
    fs = dcfg["sampling_rate"]
    X, Y, meta = load_cache(dcfg["cache_dir"], fs)
    if dcfg.get("bandpass", False):
        X = bandpass(X, fs)
    splits = make_splits(meta, dcfg)
    mean, std = stats if stats is not None else compute_norm_stats(X, splits.train)
    X -= mean[None, :, None]
    X /= std[None, :, None]
    return DataBundle(X=X, Y=Y, meta=meta, splits=splits, mean=mean, std=std, fs=fs)


def load_raw(cfg: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, Splits]:
    dcfg = cfg["data"]
    X, Y, meta = load_cache(dcfg["cache_dir"], dcfg["sampling_rate"])
    return X, Y, meta, make_splits(meta, dcfg)


def cache_info(cfg: dict) -> dict:
    path = Path(cfg["data"]["cache_dir"]) / "info.json"
    return json.loads(path.read_text()) if path.exists() else {}
