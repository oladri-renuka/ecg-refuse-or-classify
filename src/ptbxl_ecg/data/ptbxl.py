"""Build the PTB-XL cache: one float32 array of signals + labels + metadata.

    python -m ptbxl_ecg.data.ptbxl --config configs/base.yaml          # real data
    python -m ptbxl_ecg.data.ptbxl --synthetic 600 --out data/synthetic  # smoke-test data

Outputs in cache_dir:
    X_<fs>.npy   (N, 12, T) float32, raw millivolts, lead order = ptbxl_ecg.data.LEADS
    Y.npy        (N, 5) float32 multi-hot over SUPERCLASSES
    meta.csv     one row per record, row i <-> X[i]
    info.json    provenance (counts, dropped records, sampling rate)
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ptbxl_ecg.data import LEADS, NOISE_COLS, PIPELINE_VERSION, SUPERCLASSES


def load_superclass_map(root: Path) -> dict[str, str]:
    """SCP code -> diagnostic superclass, for diagnostic statements only."""
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1]
    return scp["diagnostic_class"].dropna().to_dict()


def aggregate_superclasses(scp_codes: dict[str, float], code_map: dict[str, str]) -> list[str]:
    """Same rule as the dataset's example_physionet.py: any listed diagnostic code
    (regardless of likelihood) contributes its superclass."""
    return sorted({code_map[c] for c in scp_codes if c in code_map})


def to_multihot(labels: list[str]) -> np.ndarray:
    y = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for lab in labels:
        y[SUPERCLASSES.index(lab)] = 1.0
    return y


def noise_flags(df: pd.DataFrame) -> pd.DataFrame:
    """A noise column counts as 'flagged' when it holds any non-blank annotation."""
    flags = pd.DataFrame(index=df.index)
    for col in NOISE_COLS:
        flags[f"flag_{col}"] = df[col].fillna("").astype(str).str.strip().ne("")
    flags["noisy"] = flags.any(axis=1)
    return flags


def prepare_ptbxl(root: Path, out_dir: Path, sampling_rate: int = 100) -> dict:
    import wfdb  # only needed for the real dataset
    from tqdm import tqdm

    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    code_map = load_superclass_map(root)
    db["superclasses"] = db["scp_codes"].apply(lambda d: aggregate_superclasses(d, code_map))

    n_total = len(db)
    db = db[db["superclasses"].str.len() > 0]  # benchmark protocol: drop records without a superclass
    file_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    n_samples = 10 * sampling_rate

    X = np.empty((len(db), len(LEADS), n_samples), dtype=np.float32)
    for i, rel in enumerate(tqdm(db[file_col], desc="reading WFDB")):
        signal, fields = wfdb.rdsamp(str(root / rel))
        if i == 0:
            assert [s.upper() for s in fields["sig_name"]] == [s.upper() for s in LEADS], fields["sig_name"]
        X[i] = signal.T
    np.nan_to_num(X, copy=False)

    Y = np.stack(db["superclasses"].apply(to_multihot).to_list())
    meta = pd.concat(
        [db[["patient_id", "strat_fold", "age", "sex", *NOISE_COLS]], noise_flags(db)], axis=1
    )
    meta["labels"] = db["superclasses"].apply(lambda s: "|".join(s))
    meta = meta.reset_index()  # keeps ecg_id as a column

    info = {
        "source": str(root),
        "sampling_rate": sampling_rate,
        "n_total_records": n_total,
        "n_kept": len(db),
        "n_dropped_no_superclass": n_total - len(db),
        "label_counts": dict(zip(SUPERCLASSES, Y.sum(0).astype(int).tolist())),
        "noisy_fraction": float(meta["noisy"].mean()),
    }
    _save(out_dir, X, Y, meta, info, sampling_rate)
    return info


def make_synthetic(out_dir: Path, n: int = 600, sampling_rate: int = 100, seed: int = 0) -> dict:
    """Toy ECG-like records with class-dependent morphology, for tests and smoke runs only."""
    rng = np.random.default_rng(seed)
    T = 10 * sampling_rate
    t = np.arange(T) / sampling_rate
    X = np.zeros((n, 12, T), dtype=np.float32)
    Y = np.zeros((n, 5), dtype=np.float32)
    flags = {c: np.zeros(n, dtype=bool) for c in NOISE_COLS}

    def bump(center, width, amp):
        return amp * np.exp(-0.5 * ((t[None, :] - center[:, None]) / width) ** 2).sum(0)

    for i in range(n):
        if rng.random() < 0.45:
            Y[i, 0] = 1
        else:
            Y[i, 1:] = rng.random(4) < [0.5, 0.45, 0.4, 0.25]
            if Y[i, 1:].sum() == 0:
                Y[i, 1 + rng.integers(4)] = 1
        hr = rng.uniform(50, 100) * (1.15 if Y[i, 3] else 1.0)
        beats = np.arange(rng.uniform(0, 0.5), 10, 60 / hr)
        qrs_w = 0.012 * (2.0 if Y[i, 3] else 1.0)
        qrs_a = 1.0 * (1.8 if Y[i, 4] else 1.0)
        t_a = 0.25 * (-1.0 if Y[i, 2] else 1.0)
        st = 0.15 if Y[i, 1] else 0.0
        base = (
            bump(beats - 0.16, 0.025, 0.1)
            + bump(beats - 0.02, qrs_w, -0.15)
            + bump(beats, qrs_w, qrs_a)
            + bump(beats + 0.15, 0.06, st)
            + bump(beats + 0.3, 0.05, t_a)
        )
        gains = rng.uniform(0.4, 1.2, size=12) * rng.choice([-1, 1], size=12, p=[0.2, 0.8])
        X[i] = gains[:, None] * base[None, :] + rng.normal(0, 0.01, size=(12, T))
        if rng.random() < 0.2:
            kind = NOISE_COLS[rng.integers(4)]
            flags[kind][i] = True
            if kind == "baseline_drift":
                X[i] += 0.8 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 6.3))
            elif kind == "electrodes_problems":
                X[i, rng.integers(12)] = rng.normal(0, 0.5, size=T)
            else:
                X[i] += rng.normal(0, 0.15, size=(12, T))

    meta = pd.DataFrame(
        {
            "ecg_id": np.arange(1, n + 1),
            "patient_id": np.arange(1, n + 1),
            "strat_fold": rng.integers(1, 11, size=n),
            "age": rng.integers(20, 90, size=n),
            "sex": rng.integers(0, 2, size=n),
        }
    )
    for col in NOISE_COLS:
        meta[col] = np.where(flags[col], "synthetic", None)
    meta = pd.concat([meta, noise_flags(meta)], axis=1)
    meta["labels"] = ["|".join(c for c, v in zip(SUPERCLASSES, y) if v) for y in Y]
    info = {"source": "synthetic", "sampling_rate": sampling_rate, "n_kept": n, "seed": seed}
    _save(out_dir, X, Y, meta, info, sampling_rate)
    return info


def _save(out_dir: Path, X, Y, meta, info, sampling_rate: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {"pipeline_version": PIPELINE_VERSION, **info}
    np.save(out_dir / f"X_{sampling_rate}.npy", X)
    np.save(out_dir / "Y.npy", Y)
    meta.to_csv(out_dir / "meta.csv", index=False)
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--synthetic", type=int, default=0, help="generate N synthetic records instead")
    parser.add_argument("--root", default=None, help="override data.root (PTB-XL download dir)")
    parser.add_argument("--out", default=None, help="override data.cache_dir")
    args = parser.parse_args()

    from ptbxl_ecg.utils.config import load_config

    dcfg = load_config(args.config)["data"] if Path(args.config).exists() else {"sampling_rate": 100, "root": None, "cache_dir": None}
    out = Path(args.out or dcfg["cache_dir"])
    if args.synthetic:
        info = make_synthetic(out, n=args.synthetic, sampling_rate=dcfg["sampling_rate"])
    else:
        info = prepare_ptbxl(Path(args.root or dcfg["root"]), out, dcfg["sampling_rate"])
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
