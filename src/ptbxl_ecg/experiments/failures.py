"""Failure analysis for one classifier run: where does it miss, and on what?

    python -m ptbxl_ecg.experiments.failures runs/c2_resnet1d/seed0

Writes reports/tables/failures_<name>.md with
  * per-class FN/FP counts at the frozen thresholds
  * error rates sliced by metadata (noise flag, age band, sex, number of labels)
  * the most confident misses per class (ecg_id, true labels, probability)
and reports/figures/failures_<name>.png with the 6 most confident MI misses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptbxl_ecg.data import LEADS, SUPERCLASSES
from ptbxl_ecg.data.dataset import load_raw
from ptbxl_ecg.experiments.common import FIG_DIR, TABLE_DIR, ensure_dirs
from ptbxl_ecg.metrics import MI, macro_auroc
from ptbxl_ecg.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run")
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    ensure_dirs()
    run = Path(args.run)
    preds = np.load(run / "preds.npz")
    metrics = json.loads((run / "metrics.json").read_text())
    cfg_name = metrics["name"]
    thr = np.array([metrics["test"]["thresholds"][c] for c in SUPERCLASSES])
    cfg = load_config(run / "config.yaml")
    X, _, meta, _ = load_raw(cfg)
    fs = cfg["data"]["sampling_rate"]

    idx, p, y = preds["test_idx"], preds["p_test"], preds["y_test"]
    m = meta.iloc[idx].reset_index(drop=True)
    pred = p >= thr
    lines = [f"# Failure analysis: {cfg_name} ({run.name})", ""]

    per_class = pd.DataFrame({
        "class": SUPERCLASSES,
        "positives": y.sum(0).astype(int),
        "false_negatives": ((y == 1) & ~pred).sum(0),
        "false_positives": ((y == 0) & pred).sum(0),
        "threshold": thr.round(3),
    })
    per_class["FN rate"] = (per_class["false_negatives"] / per_class["positives"]).round(3)
    lines += ["## Errors per class (frozen val thresholds)", "", per_class.to_markdown(index=False), ""]

    slices = {
        "metadata noise flag": np.where(m["noisy"], "flagged", "clean"),
        "age band": pd.cut(m["age"].clip(upper=100), [0, 40, 60, 75, 120], labels=["<40", "40-60", "60-75", "75+"]).astype(str),
        "sex (0=male,1=female)": m["sex"].astype(str),
        "# superclass labels": np.clip(y.sum(1), 1, 3).astype(int).astype(str),
    }
    rows = []
    for name, groups in slices.items():
        for g in sorted(set(groups)):
            mask = np.asarray(groups) == g
            if mask.sum() < 20:
                continue
            rows.append({"slice": name, "group": g, "n": int(mask.sum()), "macro AUROC": round(macro_auroc(y[mask], p[mask]), 4),
                         "exact-match error": round(float((pred[mask] != (y[mask] == 1)).any(1).mean()), 3)})
    lines += ["## Sliced performance", "", pd.DataFrame(rows).to_markdown(index=False), ""]

    lines += ["## Most confident misses", ""]
    for c, cls in enumerate(SUPERCLASSES):
        fn = np.flatnonzero((y[:, c] == 1) & ~pred[:, c])
        fp = np.flatnonzero((y[:, c] == 0) & pred[:, c])
        fn, fp = fn[np.argsort(p[fn, c])][: args.top], fp[np.argsort(-p[fp, c])][: args.top]
        for kind, sel in (("missed", fn), ("false alarm", fp)):
            if len(sel):
                tab = pd.DataFrame({"ecg_id": m["ecg_id"].iloc[sel].values, "true labels": m["labels"].iloc[sel].values,
                                    f"p({cls})": p[sel, c].round(3), "noisy": m["noisy"].iloc[sel].values})
                lines += [f"**{cls} {kind}**", "", tab.to_markdown(index=False), ""]
    out = TABLE_DIR / f"failures_{cfg_name}.md"
    out.write_text("\n".join(lines))

    fn = np.flatnonzero((y[:, MI] == 1) & ~pred[:, MI])
    fn = fn[np.argsort(p[fn, MI])][:6]
    if len(fn):
        fig, axes = plt.subplots(len(fn), 1, figsize=(12, 1.8 * len(fn)), sharex=True, squeeze=False)
        t = np.arange(X.shape[-1]) / fs
        for ax, k in zip(axes[:, 0], fn):
            rec = X[idx[k]]
            for j, lead in enumerate([LEADS.index(l) for l in ("II", "V2", "V5")]):
                ax.plot(t, np.clip(rec[lead], -1.5, 1.5) * 0.6 - 1.2 * j, lw=0.6, color="k")
            ax.set_yticks([0, -1.2, -2.4], ["II", "V2", "V5"], fontsize=6)
            ax.set_title(f"ecg_id {m['ecg_id'].iloc[k]}: true {m['labels'].iloc[k]}, p(MI)={p[k, MI]:.2f}, noisy={m['noisy'].iloc[k]}", fontsize=7, loc="left")
        axes[-1, 0].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"failures_{cfg_name}_mi.png", dpi=150)
    print(out.read_text())


if __name__ == "__main__":
    main()
