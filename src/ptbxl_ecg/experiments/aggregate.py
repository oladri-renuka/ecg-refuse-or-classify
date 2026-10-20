"""Collect runs/*/seed*/metrics.json into mean ± std tables (3 seeds per headline number)
and refresh the results block in README.md.

    python -m ptbxl_ecg.experiments.aggregate
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ptbxl_ecg.experiments.common import TABLE_DIR, ensure_dirs, expand_runs, fmt, load_metrics, mean_std

README = Path("README.md")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
CLF_METRICS = [("macro_auroc", "macro AUROC"), ("macro_f1", "macro F1"), ("mi_sens_at_90spec", "MI sens @90% spec")]


def main() -> None:
    ensure_dirs()
    records = [load_metrics(r) for r in expand_runs("runs/*/seed*")]
    if not records:
        print("no finished runs under runs/")
        return
    latency = pd.read_csv(TABLE_DIR / "latency.csv") if (TABLE_DIR / "latency.csv").exists() else None

    rows = []
    df = pd.DataFrame(records)
    for (exp, name), g in df[df["model"] != "autoencoder"].groupby(["experiment", "name"], sort=True):
        row = {"exp": exp, "run": name, "model": g["model"].iloc[0], "seeds": len(g)}
        for key, label in CLF_METRICS:
            row[label] = fmt(*mean_std(g["test"].apply(lambda t: t.get(key))))
        row["params (M)"] = f"{g['n_params'].iloc[0] / 1e6:.2f}" if "n_params" in g and pd.notna(g["n_params"].iloc[0]) else "-"
        if latency is not None and (latency["name"] == name).any():
            lat = latency[latency["name"] == name].iloc[-1]
            row["CPU p50 / p95 (ms)"] = f"{lat['p50_ms']:.1f} / {lat['p95_ms']:.1f}"
            row["size (MB)"] = f"{lat['size_mb']:.1f}"
        rows.append(row)
    table = pd.DataFrame(rows).fillna("-")
    table.to_csv(TABLE_DIR / "results.csv", index=False)
    md = table.to_markdown(index=False)

    ae = df[df["model"] == "autoencoder"]
    if not ae.empty:
        md += "\n\n**Quality gate (C4)**, test AUROC of reconstruction error vs. metadata noise flags: " + fmt(
            *mean_std(ae["test"].apply(lambda t: t["gate_auroc_any"])), 3
        ) + f" ({len(ae)} seeds)"
    if (TABLE_DIR / "c5_rejection.md").exists():
        md += "\n\n**Selective classification (C5)**, test macro AUROC on accepted records:\n\n" + (TABLE_DIR / "c5_rejection.md").read_text()

    (TABLE_DIR / "results.md").write_text(md)
    print(md)
    if README.exists() and START in README.read_text():
        text = README.read_text()
        README.write_text(re.sub(f"{START}.*?{END}", f"{START}\n{md}\n{END}", text, flags=re.S))
        print("README results block updated")


if __name__ == "__main__":
    main()
