import json

import numpy as np

from ptbxl_ecg.train import run


def test_classifier_end_to_end(tiny_cfg):
    res = run(tiny_cfg)
    out = tiny_cfg["output_dir"] + f"/{tiny_cfg['name']}/seed0"
    saved = json.loads(open(f"{out}/metrics.json").read())
    assert 0.0 <= saved["test"]["macro_auroc"] <= 1.0
    assert set(saved["test"]["thresholds"]) == {"NORM", "MI", "STTC", "CD", "HYP"}
    preds = np.load(f"{out}/preds.npz")
    assert preds["p_test"].shape == (len(preds["test_idx"]), 5)
    assert res["n_params"] > 0


def test_seed_reproducibility(tiny_cfg):
    tiny_cfg["train"]["epochs"] = 1
    a = run(dict(tiny_cfg))["test"]["macro_auroc"]
    b = run(dict(tiny_cfg))["test"]["macro_auroc"]
    assert abs(a - b) < 1e-6


def test_autoencoder_end_to_end(tiny_cfg):
    from conftest import ROOT
    from ptbxl_ecg.utils.config import load_config

    cfg = load_config(ROOT / "configs/autoencoder.yaml", [
        f"data.cache_dir={tiny_cfg['data']['cache_dir']}", f"output_dir={tiny_cfg['output_dir']}",
        "tracking.backend=none", "train.epochs=2", "train.batch_size=32", "train.device=cpu", "model.channels=[8,16,32]",
    ])
    res = run(cfg)
    assert res["threshold"] > 0
    assert 0.0 <= res["test"]["gate_auroc_any"] <= 1.0


def test_baseline_end_to_end(tiny_cfg):
    from conftest import ROOT
    from ptbxl_ecg.utils.config import load_config

    cfg = load_config(ROOT / "configs/baseline_logreg.yaml", [
        f"data.cache_dir={tiny_cfg['data']['cache_dir']}", f"output_dir={tiny_cfg['output_dir']}", "tracking.backend=none",
    ])
    res = run(cfg)
    assert res["test"]["macro_auroc"] > 0.6  # synthetic classes are easy from morphology features
