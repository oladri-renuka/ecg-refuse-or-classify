"""C6: how each architecture degrades under synthetic corruption of test signals,
and how often the quality gate fires at each corruption level.

    python -m ptbxl_ecg.experiments.robustness configs/c6_robustness.yaml
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

from ptbxl_ecg.data.dataset import load_raw
from ptbxl_ecg.engine import load_checkpoint
from ptbxl_ecg.eval import prepare_inputs, run_module
from ptbxl_ecg.experiments.common import FIG_DIR, TABLE_DIR, ensure_dirs, expand_runs
from ptbxl_ecg.metrics import macro_auroc
from ptbxl_ecg.utils.seed import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    args = parser.parse_args()
    rcfg = yaml.safe_load(Path(args.config).read_text())
    ensure_dirs()
    device = resolve_device("auto")

    runs = expand_runs(rcfg["clf_runs"])
    assert runs, "no finished classifier runs matched"
    X, Y, _, sp = load_raw(load_checkpoint(runs[0] / "best.pt")[1]["config"])
    X_test, Y_test = X[sp.test], Y[sp.test]
    del X

    gate = None
    if rcfg.get("ae_run") and (Path(rcfg["ae_run"]) / "best.pt").exists():
        gate = load_checkpoint(Path(rcfg["ae_run"]) / "best.pt", device)

    rows = []
    for kind, levels in rcfg["corruptions"].items():
        for level in levels:
            for run in runs:
                module, ckpt = load_checkpoint(run / "best.pt", device)
                x = prepare_inputs(X_test, ckpt, kind, level, rcfg["seed"])
                rows.append({"name": ckpt["config"]["name"], "seed": ckpt["config"]["seed"], "corruption": kind, "level": level,
                             "macro_auroc": macro_auroc(Y_test, run_module(module, x, device))})
                print(rows[-1], flush=True)
            if gate is not None:
                ae, ae_ckpt = gate
                err = run_module(ae, prepare_inputs(X_test, ae_ckpt, kind, level, rcfg["seed"]), device)
                rows.append({"name": "quality_gate", "seed": ae_ckpt["config"]["seed"], "corruption": kind, "level": level,
                             "flag_rate": float((err > ae_ckpt["threshold"]).mean())})

    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "c6_robustness.csv", index=False)

    kinds = list(rcfg["corruptions"])
    fig, axes = plt.subplots(1, len(kinds), figsize=(5 * len(kinds), 4), squeeze=False)
    clf = df[df["name"] != "quality_gate"]
    for ax, kind in zip(axes[0], kinds):
        sub = clf[clf["corruption"] == kind].groupby(["name", "level"])["macro_auroc"].agg(["mean", "std"]).reset_index()
        for name, g in sub.groupby("name"):
            ax.errorbar(g["level"], g["mean"], yerr=g["std"].fillna(0), marker="o", ms=3, capsize=2, label=name)
        if gate is not None:
            gr = df[(df["name"] == "quality_gate") & (df["corruption"] == kind)]
            ax2 = ax.twinx()
            ax2.plot(gr["level"], gr["flag_rate"], "k--", lw=1, label="gate flag rate")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("fraction rejected by gate")
        ax.set(xlabel=kind, ylabel="test macro AUROC", title=f"C6: {kind}")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "c6_robustness.png", dpi=150)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
