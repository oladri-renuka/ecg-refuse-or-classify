"""Training/inference primitives shared by train.py, eval.py and experiments."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from ptbxl_ecg.data import PIPELINE_VERSION
from ptbxl_ecg.models import build_model
from ptbxl_ecg.models.autoencoder import AEScorer
from ptbxl_ecg.models.wrappers import SlidingWindowClassifier


def autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def predict(module: torch.nn.Module, X: np.ndarray, idx: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    module.eval()
    outs = []
    for i in range(0, len(idx), batch_size):
        x = torch.from_numpy(X[idx[i : i + batch_size]]).to(device)
        outs.append(module(x).float().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0,))


def classifier_wrapper(model: torch.nn.Module, cfg: dict, total_len: int) -> SlidingWindowClassifier:
    return SlidingWindowClassifier(model, cfg["data"]["crop_len"], cfg["data"]["eval_stride"], total_len)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Returns (inference module, checkpoint dict). Classifiers come back wrapped in
    SlidingWindowClassifier (probabilities); autoencoders in AEScorer (errors)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("pipeline_version") != PIPELINE_VERSION:
        from ptbxl_ecg.data.dataset import PipelineVersionError

        raise PipelineVersionError(
            f"{path} has pipeline_version={ckpt.get('pipeline_version')}, this code is {PIPELINE_VERSION}"
        )
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    if ckpt["config"]["task"] == "autoencoder":
        module = AEScorer(model)
    else:
        module = classifier_wrapper(model, ckpt["config"], ckpt["total_len"])
    return module.to(device).eval(), ckpt
