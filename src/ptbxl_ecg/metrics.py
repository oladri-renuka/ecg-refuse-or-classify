"""Evaluation metrics. Thresholds are always chosen on validation and frozen for test."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, roc_curve

from ptbxl_ecg.data import SUPERCLASSES

MI = SUPERCLASSES.index("MI")


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        if 0 < y[:, c].sum() < len(y):
            out[c] = roc_auc_score(y[:, c], p[:, c])
    return out


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.nanmean(per_class_auroc(y, p)))


def best_f1_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    thr = np.full(y.shape[1], 0.5)
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0:
            continue
        prec, rec, cand = precision_recall_curve(y[:, c], p[:, c])
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        thr[c] = cand[np.argmax(f1)]
    return thr


def per_class_f1(y: np.ndarray, p: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return np.array([f1_score(y[:, c], p[:, c] >= thr[c], zero_division=0) for c in range(y.shape[1])])


def sensitivity_at_specificity(y: np.ndarray, p: np.ndarray, spec: float = 0.9) -> float:
    """Best TPR on the ROC curve among operating points with specificity >= spec."""
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= 1 - spec
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_specificity(y: np.ndarray, p: np.ndarray, spec: float = 0.9) -> float:
    neg = np.sort(p[y == 0])
    return float(neg[min(int(np.ceil(spec * len(neg))), len(neg) - 1)]) if len(neg) else 0.5


def sens_spec_at(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[float, float]:
    pred = p >= thr
    sens = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else float("nan")
    return sens, spec


def classification_report(
    y_test: np.ndarray, p_test: np.ndarray, y_val: np.ndarray | None = None, p_val: np.ndarray | None = None
) -> dict:
    """All headline + secondary classifier metrics for one run."""
    auc = per_class_auroc(y_test, p_test)
    report = {
        "macro_auroc": float(np.nanmean(auc)),
        "per_class_auroc": dict(zip(SUPERCLASSES, auc.round(5).tolist())),
        "mi_sens_at_90spec": sensitivity_at_specificity(y_test[:, MI], p_test[:, MI], 0.9),
    }
    if y_val is not None:
        thr = best_f1_thresholds(y_val, p_val)
        f1 = per_class_f1(y_test, p_test, thr)
        mi_thr = threshold_at_specificity(y_val[:, MI], p_val[:, MI], 0.9)
        sens, spec = sens_spec_at(y_test[:, MI], p_test[:, MI], mi_thr)
        report.update(
            macro_f1=float(f1.mean()),
            per_class_f1=dict(zip(SUPERCLASSES, f1.round(5).tolist())),
            thresholds=dict(zip(SUPERCLASSES, thr.round(6).tolist())),
            mi_frozen_threshold=mi_thr,
            mi_sens_at_frozen_thr=sens,
            mi_spec_at_frozen_thr=spec,
        )
    return report


def frozen_threshold_report(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> dict:
    """Metrics at thresholds fixed elsewhere (e.g. a checkpoint's fold-9 thresholds)."""
    report = classification_report(y, p)
    f1 = per_class_f1(y, p, np.asarray(thresholds))
    report.update(macro_f1=float(f1.mean()), per_class_f1=dict(zip(SUPERCLASSES, f1.round(5).tolist())))
    return report
