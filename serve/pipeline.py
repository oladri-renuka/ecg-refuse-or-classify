"""Serving pipeline: quality gate first, classifier only if the signal passes.

Depends only on numpy and onnxruntime (no torch, no training code), so the container
stays small. SciPy is imported only for models trained with the bandpass filter.
Preprocessing here must mirror src/data/preprocess.py; tests/test_serve.py checks that.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class SignalError(ValueError):
    pass


def _bandpass_sos(fs: float):
    try:
        from scipy.signal import butter
    except ImportError as exc:  # slim image built without scipy
        raise RuntimeError("this model was trained with bandpass=true; install scipy in the serving image") from exc
    return butter(3, [0.5, 40.0], btype="band", fs=fs, output="sos")


class _Stage:
    def __init__(self, root: Path, meta: dict, fs: float, threads: int):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(root / meta["file"]), opts, providers=["CPUExecutionProvider"])
        self.mean = np.asarray(meta["norm_mean"], dtype=np.float32)[:, None]
        self.std = np.asarray(meta["norm_std"], dtype=np.float32)[:, None]
        self.sos = _bandpass_sos(fs) if meta["bandpass"] else None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.sos is not None:
            from scipy.signal import sosfiltfilt

            x = sosfiltfilt(self.sos, x, axis=-1)
        x = ((x - self.mean) / self.std).astype(np.float32)[None]
        return self.session.run(None, {"ecg": x})[0][0]


class ECGPipeline:
    def __init__(self, artifacts_dir: str | Path = "artifacts", threads: int = 4):
        root = Path(artifacts_dir)
        self.meta = json.loads((root / "meta.json").read_text())
        self.classes: list[str] = self.meta["classes"]
        self.n_leads = len(self.meta["leads"])
        self.n_samples = int(self.meta["n_samples"])
        fs = float(self.meta["fs"])
        self.gate = _Stage(root, self.meta["gate"], fs, threads)
        self.classifier = _Stage(root, self.meta["classifier"], fs, threads)
        self.gate_threshold = float(self.meta["gate"]["threshold"])
        self.thresholds = np.asarray(self.meta["classifier"]["thresholds"], dtype=np.float32)

    def validate(self, signal) -> np.ndarray:
        x = np.asarray(signal, dtype=np.float32)
        if x.shape != (self.n_leads, self.n_samples):
            raise SignalError(f"expected shape ({self.n_leads}, {self.n_samples}) [leads x samples at {self.meta['fs']} Hz], got {x.shape}")
        if not np.isfinite(x).all():
            raise SignalError("signal contains NaN or inf")
        return x

    def predict(self, signal, force_classify: bool = False) -> dict:
        x = self.validate(signal)
        error = float(self.gate(x))
        base = {"reconstruction_error": error, "quality_threshold": self.gate_threshold}
        if error > self.gate_threshold and not force_classify:
            return {"status": "low_quality", "message": "signal quality too low", **base}
        probs = self.classifier(x)
        return {
            "status": "ok",
            "labels": [c for c, p, t in zip(self.classes, probs, self.thresholds) if p >= t],
            "probabilities": {c: round(float(p), 5) for c, p in zip(self.classes, probs)},
            "thresholds": {c: round(float(t), 5) for c, t in zip(self.classes, self.thresholds)},
            **base,
        }
