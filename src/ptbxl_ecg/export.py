"""Export the chosen classifier + quality-gate autoencoder for serving.

    python -m ptbxl_ecg.export --clf runs/c2_resnet1d/seed0/best.pt --ae runs/autoencoder/seed0/best.pt --out artifacts

Writes classifier.onnx, autoencoder.onnx, meta.json (preprocessing, thresholds)
and demo_records.npz (a handful of fold-10 records for the streaming demo and
load test). Verifies ONNX Runtime output against PyTorch before finishing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ptbxl_ecg.data import LEADS, SUPERCLASSES
from ptbxl_ecg.data.dataset import load_raw
from ptbxl_ecg.engine import load_checkpoint
from ptbxl_ecg.eval import prepare_inputs
from ptbxl_ecg.latency import export_onnx, ort_session


def _parity(module: torch.nn.Module, onnx_path: Path, x: np.ndarray) -> float:
    with torch.no_grad():
        ref = module(torch.from_numpy(x)).numpy()
    got = ort_session(onnx_path, 1).run(None, {"ecg": x})[0]
    return float(np.abs(ref - got).max())


def _prep_meta(ckpt: dict, file: str, path: str) -> dict:
    return {
        "file": file,
        "source_checkpoint": path,
        "config_name": ckpt["config"]["name"],
        "norm_mean": np.asarray(ckpt["norm_mean"]).tolist(),
        "norm_std": np.asarray(ckpt["norm_std"]).tolist(),
        "bandpass": bool(ckpt["config"]["data"].get("bandpass", False)),
    }


def export_artifacts(clf_path: str, ae_path: str, out_dir: str | Path, n_demo: int = 24) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clf, clf_ckpt = load_checkpoint(clf_path, "cpu")
    ae, ae_ckpt = load_checkpoint(ae_path, "cpu")
    assert clf_ckpt["fs"] == ae_ckpt["fs"] and clf_ckpt["total_len"] == ae_ckpt["total_len"]
    T = clf_ckpt["total_len"]

    export_onnx(clf, out / "classifier.onnx", total_len=T, output_name="probs")
    export_onnx(ae, out / "autoencoder.onnx", total_len=T, output_name="error")

    X, Y, meta, sp = load_raw(clf_ckpt["config"])
    rng = np.random.default_rng(0)
    test = sp.test
    noisy = meta["noisy"].to_numpy(bool)[test]
    pick = np.concatenate([
        rng.choice(test[~noisy], size=min(n_demo // 2, (~noisy).sum()), replace=False),
        rng.choice(test[noisy], size=min(n_demo // 2, noisy.sum()), replace=False),
    ])
    np.savez_compressed(
        out / "demo_records.npz", X=X[pick], Y=Y[pick], ecg_id=meta["ecg_id"].to_numpy()[pick],
        noisy=meta["noisy"].to_numpy(bool)[pick], classes=np.array(SUPERCLASSES),
    )

    x_clf = prepare_inputs(X[pick[:4]], clf_ckpt)
    x_ae = prepare_inputs(X[pick[:4]], ae_ckpt)
    parity = {"classifier": _parity(clf, out / "classifier.onnx", x_clf), "autoencoder": _parity(ae, out / "autoencoder.onnx", x_ae)}
    assert parity["classifier"] < 1e-4 and parity["autoencoder"] < 1e-3, f"ONNX/PyTorch mismatch: {parity}"

    info = {
        "classes": SUPERCLASSES,
        "leads": LEADS,
        "fs": clf_ckpt["fs"],
        "n_samples": T,
        "units": "mV",
        "classifier": {
            **_prep_meta(clf_ckpt, "classifier.onnx", clf_path),
            "model": clf_ckpt["config"]["model"],
            "thresholds": [float(t) for t in clf_ckpt.get("thresholds", [0.5] * len(SUPERCLASSES))],
        },
        "gate": {**_prep_meta(ae_ckpt, "autoencoder.onnx", ae_path), "threshold": float(ae_ckpt["threshold"])},
        "onnx_parity_max_abs_diff": parity,
    }
    (out / "meta.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clf", required=True)
    parser.add_argument("--ae", required=True)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    info = export_artifacts(args.clf, args.ae, args.out)
    print(json.dumps({k: v for k, v in info.items() if k in ("onnx_parity_max_abs_diff",)}, indent=2))
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
