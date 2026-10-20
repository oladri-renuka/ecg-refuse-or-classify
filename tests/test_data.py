import numpy as np
import pandas as pd
import torch

from ptbxl_ecg.data import NOISE_COLS, SUPERCLASSES
from ptbxl_ecg.data.augment import BatchAugmenter, random_crop
from ptbxl_ecg.data.dataset import load_bundle, make_splits
from ptbxl_ecg.data.preprocess import compute_norm_stats
from ptbxl_ecg.data.ptbxl import aggregate_superclasses, noise_flags, to_multihot


def test_superclass_aggregation():
    code_map = {"IMI": "MI", "ASMI": "MI", "NORM": "NORM", "LVH": "HYP"}
    # likelihood is ignored (as in example_physionet.py); non-diagnostic codes dropped
    assert aggregate_superclasses({"IMI": 15.0, "ASMI": 100.0, "SR": 0.0}, code_map) == ["MI"]
    assert aggregate_superclasses({"LVH": 50.0, "IMI": 100.0}, code_map) == ["HYP", "MI"]
    assert aggregate_superclasses({"SR": 0.0}, code_map) == []
    np.testing.assert_array_equal(to_multihot(["MI", "HYP"]), [0, 1, 0, 0, 1])
    assert SUPERCLASSES == ["NORM", "MI", "STTC", "CD", "HYP"]


def test_noise_flags_treat_blank_as_clean():
    df = pd.DataFrame({c: [None, "  ", ", I-V1,"] for c in NOISE_COLS})
    df.loc[0, "burst_noise"] = "alles"
    flags = noise_flags(df)
    assert flags["noisy"].tolist() == [True, False, True]
    assert flags["flag_static_noise"].tolist() == [False, False, True]


def test_splits_are_fold_disjoint_and_subset_is_deterministic(synthetic_cache, tiny_cfg):
    meta = pd.read_csv(synthetic_cache / "meta.csv")
    sp = make_splits(meta, tiny_cfg["data"])
    assert not set(sp.train) & set(sp.val) and not set(sp.train) & set(sp.test) and not set(sp.val) & set(sp.test)
    assert set(meta["strat_fold"].iloc[sp.test]) == {10}
    d = dict(tiny_cfg["data"], train_fraction=0.25)
    a, b = make_splits(meta, d).train, make_splits(meta, d).train
    np.testing.assert_array_equal(a, b)
    assert len(a) == round(0.25 * len(sp.train)) and set(a) <= set(sp.train)


def test_standardization_fit_on_train_only(tiny_cfg):
    bundle = load_bundle(tiny_cfg)
    tr = bundle.X[bundle.splits.train]
    np.testing.assert_allclose(tr.mean(axis=(0, 2)), 0, atol=1e-3)
    np.testing.assert_allclose(tr.std(axis=(0, 2)), 1, atol=1e-2)
    # stats must not move when only test data changes
    X = np.random.default_rng(0).normal(size=(10, 12, 50)).astype(np.float32)
    m1, _ = compute_norm_stats(X, np.arange(5))
    X[5:] += 100
    m2, _ = compute_norm_stats(X, np.arange(5))
    np.testing.assert_allclose(m1, m2)


def test_crop_and_augment_shapes():
    torch.manual_seed(0)
    x = torch.arange(2 * 12 * 1000, dtype=torch.float32).view(2, 12, 1000)
    c = random_crop(x, 250)
    assert c.shape == (2, 12, 250)
    # crops are contiguous windows of the original
    start = int(c[0, 0, 0].item()) % 1000
    torch.testing.assert_close(c[0], x[0, :, start : start + 250])
    aug = BatchAugmenter({"amp_scale": [0.9, 1.1], "noise_p": 1.0, "noise_sigma": [0.01, 0.05], "wander_p": 1.0,
                          "wander_freq_hz": [0.05, 0.5], "wander_amp": [0, 0.3], "lead_dropout_p": 1.0}, fs=100)
    out = aug(torch.randn(4, 12, 250))
    assert out.shape == (4, 12, 250)
    assert (out.abs().sum(-1) > 0).any(1).all(), "at least one lead must survive dropout"
