import numpy as np

from ptbxl_ecg.metrics import best_f1_thresholds, classification_report, per_class_f1, sensitivity_at_specificity, threshold_at_specificity


def test_perfect_and_random_auroc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(500, 5)).astype(float)
    rep = classification_report(y, y * 0.8 + 0.1, y, y * 0.8 + 0.1)
    assert rep["macro_auroc"] == 1.0 and rep["macro_f1"] == 1.0
    assert abs(classification_report(y, rng.random((500, 5)))["macro_auroc"] - 0.5) < 0.08


def test_f1_threshold_is_optimal_on_its_own_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=(400, 2)).astype(float)
    p = np.clip(y * 0.3 + rng.random((400, 2)) * 0.7, 0, 1)
    thr = best_f1_thresholds(y, p)
    best = per_class_f1(y, p, thr)
    for t in np.linspace(0.05, 0.95, 19):
        assert (per_class_f1(y, p, np.full(2, t)) <= best + 1e-9).all()


def test_sensitivity_at_specificity():
    y = np.array([0] * 10 + [1] * 10)
    p = np.r_[np.linspace(0, 0.5, 10), np.linspace(0.45, 1, 10)]
    s = sensitivity_at_specificity(y, p, 0.9)
    assert 0.8 <= s <= 1.0
    thr = threshold_at_specificity(y, p, 0.9)
    assert ((p[y == 0] < thr).mean()) >= 0.9
