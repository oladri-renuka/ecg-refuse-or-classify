"""C4: does autoencoder reconstruction error detect metadata-flagged noisy records?
C5: does rejecting the highest-error records improve accuracy on the rest?

    python -m ptbxl_ecg.experiments.quality_gate configs/c4c5_quality_gate.yaml

Controls: random rejection at the same coverage, and rejection by the metadata
flags themselves (an 'oracle' the deployed system does not have). Also reports
classifier AUROC on flagged vs clean test records: if noise doesn't hurt the
classifier, no gate can help, and we want to know that.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from ptbxl_ecg.data import NOISE_COLS
from ptbxl_ecg.data.dataset import load_raw
from ptbxl_ecg.experiments.common import FIG_DIR, TABLE_DIR, ensure_dirs, expand_runs, fmt, mean_std
from ptbxl_ecg.metrics import macro_auroc
from ptbxl_ecg.utils.config import load_config


def _align(clf_preds, ae_errors) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match classifier test predictions and AE test errors by cache index."""
    pos = {i: k for k, i in enumerate(ae_errors["test_idx"])}
    order = np.array([pos[i] for i in clf_preds["test_idx"]])
    return clf_preds["test_idx"], clf_preds["p_test"], clf_preds["y_test"], ae_errors["err_test"][order]


def reject_by_score(y, p, score, frac):
    keep = np.argsort(score)[: len(score) - int(round(frac * len(score)))]
    return macro_auroc(y[keep], p[keep])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    args = parser.parse_args()
    gcfg = yaml.safe_load(Path(args.config).read_text())
    ensure_dirs()

    ae_runs, clf_runs = expand_runs(gcfg["ae_runs"]), expand_runs(gcfg["clf_runs"])
    assert ae_runs and clf_runs, "no finished runs matched; train the autoencoder and classifier first"
    _, _, meta, _ = load_raw(load_config(ae_runs[0] / "config.yaml"))
    flags = {c: meta[f"flag_{c}"].to_numpy(bool) for c in NOISE_COLS}
    noisy = meta["noisy"].to_numpy(bool)

    # ---- C4: gate AUROC per AE seed -----------------------------------------
    c4 = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for run in ae_runs:
        e = np.load(run / "errors.npz")
        idx, err = e["test_idx"], e["err_test"]
        row = {"run": str(run), "any": roc_auc_score(noisy[idx], err)}
        for c in NOISE_COLS:
            m = flags[c][idx]
            row[c] = roc_auc_score(m, err) if 0 < m.sum() < len(m) else np.nan
        row["flag_rate_at_threshold"] = float((err > float(e["threshold"])).mean())
        row["recall_noisy_at_threshold"] = float((err[noisy[idx]] > float(e["threshold"])).mean())
        c4.append(row)
        fpr, tpr, _ = roc_curve(noisy[idx], err)
        ax[0].plot(fpr, tpr, lw=1, label=f"{run.name} AUROC {row['any']:.3f}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set(xlabel="FPR", ylabel="TPR", title="C4: reconstruction error vs. metadata noise flag")
    ax[0].legend(fontsize=8)
    e0 = np.load(ae_runs[0] / "errors.npz")
    bins = np.logspace(np.log10(e0["err_test"].min()), np.log10(e0["err_test"].max()), 60)
    ax[1].hist(e0["err_test"][~e0["noisy_test"]], bins=bins, alpha=0.6, density=True, label="clean (metadata)")
    ax[1].hist(e0["err_test"][e0["noisy_test"]], bins=bins, alpha=0.6, density=True, label="flagged noisy")
    ax[1].axvline(float(e0["threshold"]), color="k", ls="--", lw=1, label="deployed threshold")
    ax[1].set(xscale="log", xlabel="reconstruction MSE (test)", title="Error distribution")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c4_quality_gate.png", dpi=150)
    c4df = pd.DataFrame(c4)
    c4df.to_csv(TABLE_DIR / "c4_gate_auroc.csv", index=False)

    # ---- C5: selective classification ----------------------------------------
    rng = np.random.default_rng(0)
    fracs = gcfg["reject_fracs"]
    curve_fracs = np.linspace(0, gcfg["curve_max_reject"], 17)
    rows, curves = [], {"autoencoder": [], "random": [], "metadata_flags": []}
    for k, clf_run in enumerate(clf_runs):
        ae = np.load(ae_runs[k % len(ae_runs)] / "errors.npz")  # pair seeds when counts match
        idx, p, y, err = _align(np.load(clf_run / "preds.npz"), ae)
        flag_score = noisy[idx].astype(float) + 1e-3 * rng.random(len(idx))  # flagged first, random within
        sub = {"clean": ~noisy[idx], "flagged": noisy[idx]}
        for f in fracs:
            rnd = np.mean([reject_by_score(y, p, rng.random(len(idx)), f) for _ in range(gcfg["n_random_draws"])])
            rows.append({
                "clf_run": str(clf_run), "reject_frac": f, "coverage": 1 - f,
                "auroc_autoencoder": reject_by_score(y, p, err, f),
                "auroc_random": rnd,
                "auroc_metadata_flags": reject_by_score(y, p, flag_score, f),
                "auroc_on_clean_subset": macro_auroc(y[sub["clean"]], p[sub["clean"]]),
                "auroc_on_flagged_subset": macro_auroc(y[sub["flagged"]], p[sub["flagged"]]),
            })
        curves["autoencoder"].append([reject_by_score(y, p, err, f) for f in curve_fracs])
        curves["random"].append([np.mean([reject_by_score(y, p, rng.random(len(idx)), f) for _ in range(10)]) for f in curve_fracs])
        curves["metadata_flags"].append([reject_by_score(y, p, flag_score, f) for f in curve_fracs])

    c5 = pd.DataFrame(rows)
    c5.to_csv(TABLE_DIR / "c5_rejection.csv", index=False)
    summary = []
    for f, g in c5.groupby("reject_frac"):
        summary.append({
            "reject": f"{f:.0%}", "coverage": f"{1 - f:.0%}",
            **{name: fmt(*mean_std(g[f"auroc_{name}"])) for name in ("autoencoder", "random", "metadata_flags")},
        })
    (TABLE_DIR / "c5_rejection.md").write_text(pd.DataFrame(summary).to_markdown(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, style in [("autoencoder", "-"), ("random", ":"), ("metadata_flags", "--")]:
        arr = np.asarray(curves[name])
        ax.plot(1 - curve_fracs, arr.mean(0), style, label=name.replace("_", " "))
        ax.fill_between(1 - curve_fracs, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), alpha=0.15)
    ax.invert_xaxis()
    ax.set(xlabel="coverage (fraction of test records classified)", ylabel="macro AUROC on accepted records", title="C5: coverage vs. accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c5_coverage_accuracy.png", dpi=150)

    print("C4 gate AUROC (test, vs any metadata noise flag):", fmt(*mean_std(c4df["any"])))
    print(c4df.to_string(index=False))
    print((TABLE_DIR / "c5_rejection.md").read_text())
    print("classifier AUROC clean vs flagged subsets:", fmt(*mean_std(c5["auroc_on_clean_subset"])), "vs", fmt(*mean_std(c5["auroc_on_flagged_subset"])))


if __name__ == "__main__":
    main()
