"""The surface Project 3 imports. If one of these breaks, bump the major version."""
import json

import numpy as np
import pytest
import torch


def test_public_names_exist():
    import ptbxl_ecg

    for name in ptbxl_ecg.__all__:
        assert hasattr(ptbxl_ecg, name), name


def test_default_split_matches_config_split(synthetic_cache, tiny_cfg):
    from ptbxl_ecg import load_cache, make_splits

    _, _, meta = load_cache(synthetic_cache)
    a, b = make_splits(meta), make_splits(meta, tiny_cfg["data"])
    for split in ("train", "val", "test"):
        np.testing.assert_array_equal(getattr(a, split), getattr(b, split))


def test_cache_version_guard(synthetic_cache, tmp_path):
    import shutil

    from ptbxl_ecg import PipelineVersionError, load_cache

    stale = tmp_path / "stale"
    shutil.copytree(synthetic_cache, stale)
    info = json.loads((stale / "info.json").read_text())
    info["pipeline_version"] = -1
    (stale / "info.json").write_text(json.dumps(info))
    with pytest.raises(PipelineVersionError):
        load_cache(stale)


def test_evaluator_matches_training_metrics(tiny_cfg):
    """Evaluator on raw mV must reproduce the test AUROC written by training."""
    from ptbxl_ecg import Evaluator, load_cache, make_splits
    from ptbxl_ecg.train import run

    res = run(tiny_cfg)
    ckpt = f"{tiny_cfg['output_dir']}/{tiny_cfg['name']}/seed0/best.pt"
    ev = Evaluator(ckpt)
    X, Y, meta = load_cache(tiny_cfg["data"]["cache_dir"])
    test = make_splits(meta).test
    rep = ev.evaluate(X[test], Y[test])
    assert abs(rep["macro_auroc"] - res["test"]["macro_auroc"]) < 1e-6
    assert ev.predict_proba(X[test[:3]]).shape == (3, 5)
    with pytest.raises(ValueError):
        ev.predict_proba(X[test[:3], :, :500])


def test_checkpoint_version_guard(tiny_cfg, tmp_path):
    from ptbxl_ecg import PipelineVersionError, load_checkpoint
    from ptbxl_ecg.train import run

    tiny_cfg["train"]["epochs"] = 1
    run(tiny_cfg)
    path = f"{tiny_cfg['output_dir']}/{tiny_cfg['name']}/seed0/best.pt"
    ckpt = torch.load(path, weights_only=False)
    ckpt["pipeline_version"] = -1
    torch.save(ckpt, tmp_path / "old.pt")
    with pytest.raises(PipelineVersionError):
        load_checkpoint(tmp_path / "old.pt")
