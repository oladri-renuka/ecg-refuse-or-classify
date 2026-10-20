from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

FIG_DIR = Path("reports/figures")
TABLE_DIR = Path("reports/tables")


def expand_runs(patterns: str | list[str]) -> list[Path]:
    patterns = [patterns] if isinstance(patterns, str) else patterns
    runs = sorted({Path(p) for pat in patterns for p in glob.glob(pat) if (Path(p) / "metrics.json").exists()})
    return runs


def load_metrics(run: Path) -> dict:
    return json.loads((run / "metrics.json").read_text())


def mean_std(values) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def fmt(mean: float, std: float, digits: int = 3) -> str:
    return "n/a" if np.isnan(mean) else f"{mean:.{digits}f} ± {std:.{digits}f}"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
