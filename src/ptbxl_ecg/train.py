"""Train one experiment from a config file, for one or more seeds.

    python -m ptbxl_ecg.train configs/c1_resnet1d.yaml --seeds 0 1 2
    python -m ptbxl_ecg.train configs/c3_bilstm.yaml --seeds 0 1 2 --set data.train_fraction=0.25

Each run writes runs/<name>/seed<k>/{best.pt, metrics.json, preds.npz, config.yaml}.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ptbxl_ecg import __version__
from ptbxl_ecg.data import NOISE_COLS, PIPELINE_VERSION
from ptbxl_ecg.data.augment import BatchAugmenter, random_crop
from ptbxl_ecg.data.dataset import DataBundle, load_bundle
from ptbxl_ecg.engine import autocast, classifier_wrapper, predict
from ptbxl_ecg.metrics import classification_report, macro_auroc
from ptbxl_ecg.models import build_model, count_params
from ptbxl_ecg.models.autoencoder import AEScorer
from ptbxl_ecg.utils.config import dump_config, load_config
from ptbxl_ecg.utils.seed import resolve_device, set_seed
from ptbxl_ecg.utils.tracking import Tracker


def run_dir_for(cfg: dict) -> Path:
    return Path(cfg["output_dir"]) / cfg["name"] / f"seed{cfg['seed']}"


def _batches(idx: np.ndarray, batch_size: int, shuffle: bool):
    order = idx[torch.randperm(len(idx)).numpy()] if shuffle else idx
    n_steps = max(1, len(order) // batch_size)  # drop the ragged last batch
    for i in range(n_steps):
        yield order[i * batch_size : (i + 1) * batch_size]


def _save_ckpt(path: Path, model: nn.Module, cfg: dict, bundle: DataBundle, **extra) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "norm_mean": bundle.mean,
            "norm_std": bundle.std,
            "fs": bundle.fs,
            "total_len": bundle.X.shape[-1],
            "pipeline_version": PIPELINE_VERSION,
            "package_version": __version__,
            **extra,
        },
        path,
    )


def train_classifier(cfg: dict, bundle: DataBundle, tracker: Tracker, out: Path) -> dict:
    tcfg, dcfg = cfg["train"], cfg["data"]
    device = resolve_device(tcfg["device"])
    model = build_model(cfg["model"]).to(device)
    wrapper = classifier_wrapper(model, cfg, bundle.X.shape[-1])
    augment = BatchAugmenter(cfg.get("augment"), bundle.fs)
    sp = bundle.splits

    pos_weight = None
    if tcfg.get("pos_weight"):
        pos = bundle.Y[sp.train].sum(0)
        pos_weight = torch.tensor((len(sp.train) - pos) / np.clip(pos, 1, None), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    steps = max(1, len(sp.train) // tcfg["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tcfg["lr"], epochs=tcfg["epochs"], steps_per_epoch=steps)

    best_auc, best_epoch, bad = -np.inf, -1, 0
    t0 = time.time()
    for epoch in range(tcfg["epochs"]):
        model.train()
        losses = []
        for b in _batches(sp.train, tcfg["batch_size"], shuffle=True):
            x = random_crop(torch.from_numpy(bundle.X[b]), dcfg["crop_len"]).to(device, non_blocking=True)
            y = torch.from_numpy(bundle.Y[b]).to(device, non_blocking=True)
            x = augment(x)
            with autocast(device, tcfg.get("amp", False)):
                loss = criterion(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        val_probs = predict(wrapper, bundle.X, sp.val, device, tcfg["eval_batch_size"])
        val_auc = macro_auroc(bundle.Y[sp.val], val_probs)
        tracker.log_metrics({"train_loss": float(np.mean(losses)), "val_macro_auroc": val_auc, "lr": sched.get_last_lr()[0]}, step=epoch)
        print(f"[{cfg['name']} s{cfg['seed']}] epoch {epoch:3d} loss {np.mean(losses):.4f} val_auroc {val_auc:.4f}", flush=True)

        if val_auc > best_auc:
            best_auc, best_epoch, bad = val_auc, epoch, 0
            _save_ckpt(out / "best.pt", model, cfg, bundle, epoch=epoch, val_macro_auroc=val_auc)
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                print(f"early stopping at epoch {epoch} (best {best_epoch})")
                break

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    p_val = predict(wrapper, bundle.X, sp.val, device, tcfg["eval_batch_size"])
    p_test = predict(wrapper, bundle.X, sp.test, device, tcfg["eval_batch_size"])
    y_val, y_test = bundle.Y[sp.val], bundle.Y[sp.test]
    test_report = classification_report(y_test, p_test, y_val, p_val)
    ckpt["thresholds"] = list(test_report["thresholds"].values())
    torch.save(ckpt, out / "best.pt")
    np.savez_compressed(out / "preds.npz", val_idx=sp.val, test_idx=sp.test, p_val=p_val, p_test=p_test, y_val=y_val, y_test=y_test)

    return {
        "model": cfg["model"]["name"],
        "n_params": count_params(model),
        "best_epoch": best_epoch,
        "train_size": int(len(sp.train)),
        "train_time_s": time.time() - t0,
        "val": {"macro_auroc": macro_auroc(y_val, p_val)},
        "test": test_report,
    }


def train_autoencoder(cfg: dict, bundle: DataBundle, tracker: Tracker, out: Path) -> dict:
    from sklearn.metrics import roc_auc_score

    tcfg = cfg["train"]
    device = resolve_device(tcfg["device"])
    noisy = bundle.meta["noisy"].to_numpy(bool)
    sp = bundle.splits
    train_idx = sp.train[~noisy[sp.train]]      # clean training records only
    val_clean = sp.val[~noisy[sp.val]]

    model = build_model(cfg["model"]).to(device)
    scorer = AEScorer(model)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    best, best_epoch, bad = np.inf, -1, 0
    t0 = time.time()
    for epoch in range(tcfg["epochs"]):
        model.train()
        losses = []
        for b in _batches(train_idx, tcfg["batch_size"], shuffle=True):
            x = torch.from_numpy(bundle.X[b]).to(device)
            loss = nn.functional.mse_loss(model(x), x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val_loss = float(predict(scorer, bundle.X, val_clean, device, tcfg["eval_batch_size"]).mean())
        tracker.log_metrics({"train_mse": float(np.mean(losses)), "val_clean_mse": val_loss}, step=epoch)
        print(f"[{cfg['name']} s{cfg['seed']}] epoch {epoch:3d} train_mse {np.mean(losses):.4f} val_clean_mse {val_loss:.4f}", flush=True)
        if val_loss < best:
            best, best_epoch, bad = val_loss, epoch, 0
            _save_ckpt(out / "best.pt", model, cfg, bundle, epoch=epoch, val_clean_mse=val_loss)
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    err_val = predict(scorer, bundle.X, sp.val, device, tcfg["eval_batch_size"])
    err_test = predict(scorer, bundle.X, sp.test, device, tcfg["eval_batch_size"])
    threshold = float(np.percentile(err_val[~noisy[sp.val]], cfg["gate"]["clean_percentile"]))
    ckpt["threshold"] = threshold
    torch.save(ckpt, out / "best.pt")

    flags = {c: bundle.meta[f"flag_{c}"].to_numpy(bool) for c in NOISE_COLS}
    np.savez_compressed(
        out / "errors.npz", val_idx=sp.val, test_idx=sp.test, err_val=err_val, err_test=err_test,
        noisy_val=noisy[sp.val], noisy_test=noisy[sp.test], threshold=threshold,
    )

    def auc(mask, err):
        return float(roc_auc_score(mask, err)) if 0 < mask.sum() < len(mask) else float("nan")

    test_noisy = noisy[sp.test]
    return {
        "model": "autoencoder",
        "n_params": count_params(model),
        "best_epoch": best_epoch,
        "train_size": int(len(train_idx)),
        "train_time_s": time.time() - t0,
        "threshold": threshold,
        "val": {"clean_mse": best, "gate_auroc_any": auc(noisy[sp.val], err_val)},
        "test": {
            "gate_auroc_any": auc(test_noisy, err_test),
            **{f"gate_auroc_{c}": auc(flags[c][sp.test], err_test) for c in NOISE_COLS},
            "flag_rate_at_threshold": float((err_test > threshold).mean()),
            "noisy_fraction": float(test_noisy.mean()),
        },
    }


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    out = run_dir_for(cfg)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out / "config.yaml")
    tracker = Tracker(cfg, f"{cfg['name']}-seed{cfg['seed']}")
    try:
        if cfg["task"] == "baseline":
            from ptbxl_ecg.baseline import train_baseline

            results = train_baseline(cfg, out)
        else:
            bundle = load_bundle(cfg)
            fn = train_autoencoder if cfg["task"] == "autoencoder" else train_classifier
            results = fn(cfg, bundle, tracker, out)
        results.update(name=cfg["name"], experiment=cfg.get("experiment"), seed=cfg["seed"], config=cfg["_config_path"])
        (out / "metrics.json").write_text(json.dumps(results, indent=2))
        tracker.log_metrics({"final": {"val": results["val"], "test": results["test"]}})
        tracker.log_artifact(str(out / "metrics.json"))
    finally:
        tracker.finish()
    print(json.dumps({"name": cfg["name"], "seed": cfg["seed"], "test": results["test"]}, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], help="dotted.key=value overrides")
    args = parser.parse_args()
    base = load_config(args.config, args.overrides)
    for seed in args.seeds if args.seeds is not None else [base["seed"]]:
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        run(cfg)


if __name__ == "__main__":
    main()
