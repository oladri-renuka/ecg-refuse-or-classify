"""1D Grad-CAM for the MI output of a ResNet1d on 10 true-positive MI test cases.

    python -m ptbxl_ecg.experiments.gradcam runs/c1_resnet1d/seed0/best.pt --stage 2

Beyond the overlay figure, quantifies whether attention falls on ST segments:
enrichment = (CAM mass inside ST windows) / (fraction of time inside ST windows),
with ST windows taken as R+60 ms .. R+240 ms from a multi-lead R-peak detector.
Enrichment > 1 means the model looks at ST segments more than chance.
CAM is computed on the full 10 s record (the network is fully convolutional).
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ptbxl_ecg.baseline import detect_r_peaks
from ptbxl_ecg.data import LEADS
from ptbxl_ecg.data.dataset import load_raw
from ptbxl_ecg.data.preprocess import bandpass
from ptbxl_ecg.engine import load_checkpoint
from ptbxl_ecg.eval import prepare_inputs, run_module
from ptbxl_ecg.experiments.common import FIG_DIR, TABLE_DIR, ensure_dirs
from ptbxl_ecg.metrics import MI


def gradcam(model, x: torch.Tensor, stage: int, cls: int) -> np.ndarray:
    feats = {}
    handle = model.stages[stage].register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    model.zero_grad()
    out = model(x)
    act = feats["a"]
    grad = torch.autograd.grad(out[:, cls].sum(), act)[0]
    handle.remove()
    cam = F.relu((grad.mean(-1, keepdim=True) * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[:, 0]
    cam = cam / cam.amax(-1, keepdim=True).clamp_min(1e-8)
    return cam.detach().numpy()


def st_mask(x_raw: np.ndarray, fs: float) -> np.ndarray:
    mask = np.zeros(x_raw.shape[-1], dtype=bool)
    for r in detect_r_peaks(bandpass(x_raw, fs), fs):
        mask[r + int(0.06 * fs) : r + int(0.24 * fs)] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--stage", type=int, default=2, help="ResNet stage to hook (higher = coarser)")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    ensure_dirs()

    wrapper, ckpt = load_checkpoint(args.checkpoint, "cpu")
    assert ckpt["config"]["model"]["name"] == "resnet1d", "Grad-CAM is implemented for the ResNet1d"
    model = wrapper.model
    fs = ckpt["fs"]
    X, Y, meta, sp = load_raw(ckpt["config"])
    test = sp.test
    probs = run_module(wrapper, prepare_inputs(X[test], ckpt), "cpu")
    tp = np.flatnonzero((Y[test, MI] == 1) & (probs[:, MI] >= ckpt.get("thresholds", [0.5] * 5)[MI]))
    order = tp[np.argsort(-probs[tp, MI])][: args.n]
    chosen = test[order]

    x = torch.from_numpy(prepare_inputs(X[chosen], ckpt)).requires_grad_(False)
    cams = gradcam(model, x, args.stage, MI)

    results, show_leads = [], [LEADS.index("II"), LEADS.index("V2"), LEADS.index("V5")]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 1.6 * len(chosen)), sharex=True)
    t = np.arange(X.shape[-1]) / fs
    for k, (i, cam) in enumerate(zip(chosen, cams)):
        mask = st_mask(X[i], fs)
        enrich = float(cam[mask].sum() / cam.sum() / mask.mean()) if mask.any() and cam.sum() > 0 else float("nan")
        results.append({"ecg_id": int(meta["ecg_id"][i]), "labels": meta["labels"][i], "p_mi": float(probs[order[k], MI]),
                        "st_fraction_of_time": float(mask.mean()), "st_enrichment": enrich})
        ax = axes[k] if len(chosen) > 1 else axes
        ax.imshow(cam[None, :], aspect="auto", cmap="Reds", alpha=0.5, extent=[0, t[-1], -2, 2], vmin=0, vmax=1)
        for j, lead in enumerate(show_leads):
            ax.plot(t, np.clip(X[i, lead], -1.5, 1.5) * 0.6 + (1.2 - 1.2 * j), lw=0.6, color="k")
        ax.fill_between(t, -2, 2, where=mask, color="tab:blue", alpha=0.08, step="mid")
        ax.set_yticks([1.2, 0.0, -1.2], [LEADS[l] for l in show_leads], fontsize=6)
        ax.set_title(f"ecg_id {results[-1]['ecg_id']} ({results[-1]['labels']}), ST enrichment {enrich:.2f}", fontsize=7, loc="left")
    axes[-1].set_xlabel("time (s); red = Grad-CAM for MI, blue band = estimated ST segment")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gradcam_mi.png", dpi=150)
    summary = {"stage": args.stage, "mean_st_enrichment": float(np.nanmean([r["st_enrichment"] for r in results])), "cases": results}
    (TABLE_DIR / "gradcam_mi.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
