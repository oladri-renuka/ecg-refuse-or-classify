"""Evaluate a trained checkpoint on a split, optionally under a synthetic corruption.

    python -m ptbxl_ecg.eval runs/c1_resnet1d/seed0/best.pt
    python -m ptbxl_ecg.eval runs/c1_resnet1d/seed0/best.pt --corrupt baseline_wander_mv=0.5

Preprocessing (bandpass flag, standardization stats) always comes from the
checkpoint, so evaluation can never refit anything on test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ptbxl_ecg.data import corruptions
from ptbxl_ecg.data.dataset import load_raw
from ptbxl_ecg.data.preprocess import preprocess
from ptbxl_ecg.engine import load_checkpoint, predict
from ptbxl_ecg.metrics import classification_report
from ptbxl_ecg.utils.seed import resolve_device


def prepare_inputs(X_raw: np.ndarray, ckpt: dict, kind: str | None = None, level: float = 0.0, seed: int = 0) -> np.ndarray:
    """Raw mV records -> (optionally corrupted) -> checkpoint's preprocessing."""
    fs = ckpt["fs"]
    x = X_raw
    if kind is not None and level:
        x = corruptions.apply(kind, X_raw.copy(), level, fs, np.random.default_rng(seed))
    return preprocess(x, ckpt["norm_mean"], ckpt["norm_std"], fs, ckpt["config"]["data"].get("bandpass", False))


def run_module(module, X: np.ndarray, device, batch_size: int = 256) -> np.ndarray:
    return predict(module, X, np.arange(len(X)), device, batch_size)


def evaluate(ckpt_path: str, split: str = "test", kind: str | None = None, level: float = 0.0, seed: int = 0) -> dict:
    device = resolve_device("auto")
    module, ckpt = load_checkpoint(ckpt_path, device)
    X, Y, _, sp = load_raw(ckpt["config"])
    idx = getattr(sp, split)
    probs = run_module(module, prepare_inputs(X[idx], ckpt, kind, level, seed), device)
    report = classification_report(Y[idx], probs)
    if "thresholds" in ckpt:
        from ptbxl_ecg.metrics import per_class_f1

        report["macro_f1_frozen"] = float(per_class_f1(Y[idx], probs, np.asarray(ckpt["thresholds"])).mean())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--corrupt", default=None, help="kind=level, e.g. gaussian_mv=0.1")
    parser.add_argument("--out", default=None, help="write JSON here")
    args = parser.parse_args()
    kind, level = (None, 0.0)
    if args.corrupt:
        kind, _, lv = args.corrupt.partition("=")
        level = float(lv)
    report = evaluate(args.checkpoint, args.split, kind, level)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
