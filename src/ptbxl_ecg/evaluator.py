"""Frozen-classifier evaluator: the supported entry point for downstream projects.

    from ptbxl_ecg import Evaluator, load_cache, make_splits
    ev = Evaluator("best.pt")                     # a checkpoint trained by this repo
    X, Y, meta = load_cache("data/processed")
    test = make_splits(meta).test                 # fold 10, same split as here
    ev.evaluate(X[test], Y[test])                 # macro AUROC, per-class, F1 at frozen thresholds
    ev.predict_proba(X_synthetic_mV)              # (N, 5) for any raw 12 x 1000 mV records

Inputs are raw millivolts. Preprocessing (bandpass flag, train-fold standardization)
always comes from the checkpoint, so every caller scores signals exactly as this
project's test numbers were produced.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ptbxl_ecg.data import LEADS, SUPERCLASSES
from ptbxl_ecg.data.preprocess import preprocess
from ptbxl_ecg.engine import load_checkpoint, predict
from ptbxl_ecg.metrics import frozen_threshold_report


class Evaluator:
    def __init__(self, checkpoint: str | Path, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.module, self.ckpt = load_checkpoint(checkpoint, self.device)
        if self.ckpt["config"]["task"] != "classifier":
            raise ValueError(f"{checkpoint} is a {self.ckpt['config']['task']} checkpoint, not a classifier")
        self.checkpoint = str(checkpoint)
        self.classes = list(SUPERCLASSES)
        self.leads = list(LEADS)
        self.fs: int = self.ckpt["fs"]
        self.n_samples: int = self.ckpt["total_len"]
        self.thresholds = np.asarray(self.ckpt.get("thresholds", [0.5] * len(SUPERCLASSES)), dtype=np.float32)
        self.norm_mean = np.asarray(self.ckpt["norm_mean"])
        self.norm_std = np.asarray(self.ckpt["norm_std"])
        self.bandpass = bool(self.ckpt["config"]["data"].get("bandpass", False))

    def preprocess(self, X_mv: np.ndarray) -> np.ndarray:
        X_mv = np.asarray(X_mv, dtype=np.float32)
        if X_mv.ndim != 3 or X_mv.shape[1:] != (len(self.leads), self.n_samples):
            raise ValueError(f"expected (N, {len(self.leads)}, {self.n_samples}) mV at {self.fs} Hz, got {X_mv.shape}")
        return preprocess(X_mv, self.norm_mean, self.norm_std, self.fs, self.bandpass)

    def predict_proba(self, X_mv: np.ndarray, batch_size: int = 256) -> np.ndarray:
        X = self.preprocess(X_mv)
        return predict(self.module, X, np.arange(len(X)), self.device, batch_size)

    def predict_labels(self, X_mv: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_mv) >= self.thresholds

    def evaluate(self, X_mv: np.ndarray, Y: np.ndarray) -> dict:
        return frozen_threshold_report(np.asarray(Y), self.predict_proba(X_mv), self.thresholds)

    def info(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "config_name": self.ckpt["config"]["name"],
            "model": self.ckpt["config"]["model"]["name"],
            "pipeline_version": self.ckpt["pipeline_version"],
            "package_version": self.ckpt.get("package_version"),
            "fs": self.fs,
            "n_samples": self.n_samples,
            "thresholds": dict(zip(self.classes, self.thresholds.round(5).tolist())),
        }
