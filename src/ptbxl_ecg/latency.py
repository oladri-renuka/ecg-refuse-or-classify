"""ONNX export + CPU latency benchmark (batch 1, ONNX Runtime, N threads).

    python -m ptbxl_ecg.latency runs/c1_*/seed0/config.yaml runs/c2_*/seed0/config.yaml   # -> reports/tables/latency.csv
    python -m ptbxl_ecg.latency --artifacts artifacts                              # deployed pipeline, end to end

Latency is per 10-second record and includes every sliding window, i.e. the
exact graph the API runs. Measure on the deployment-class CPU, not the GPU box.
Inputs may be checkpoints (best.pt) or a run's resolved config.yaml: latency does
not depend on the weights, so the architecture alone is timed when no checkpoint
is at hand (e.g. results synced without the large .pt files).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def export_onnx(module: torch.nn.Module, path: str | Path, n_leads: int = 12, total_len: int = 1000, output_name: str = "output") -> None:
    module = module.cpu().eval()
    dummy = torch.randn(1, n_leads, total_len)
    torch.onnx.export(
        module, (dummy,), str(path), input_names=["ecg"], output_names=[output_name],
        dynamic_axes={"ecg": {0: "batch"}, output_name: {0: "batch"}}, opset_version=17, dynamo=False,
    )


def ort_session(path: str | Path, threads: int = 4):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def time_fn(fn, n_warmup: int = 20, n_runs: int = 200) -> dict:
    for _ in range(n_warmup):
        fn()
    times = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1e3
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(times.mean())}


def benchmark_onnx(path: str | Path, threads: int = 4, n_warmup: int = 20, n_runs: int = 200, shape=(1, 12, 1000)) -> dict:
    sess = ort_session(path, threads)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    stats = time_fn(lambda: sess.run(None, {"ecg": x}), n_warmup, n_runs)
    stats["size_mb"] = os.path.getsize(path) / 1e6
    return stats


def benchmark_module(module: torch.nn.Module, threads: int = 4, n_warmup: int = 20, n_runs: int = 200, total_len: int = 1000) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        export_onnx(module, path, total_len=total_len)
        return benchmark_onnx(path, threads, n_warmup, n_runs, (1, 12, total_len))


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == "Darwin":
            import subprocess

            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or cpu
        elif Path("/proc/cpuinfo").exists():
            cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return f"{cpu} / {platform.system()} / {os.cpu_count()} cores"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="best.pt checkpoints or run config.yaml files")
    parser.add_argument("--artifacts", default=None, help="benchmark the exported serving pipeline instead")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--runs", dest="n_runs", type=int, default=200, help="timed iterations")
    parser.add_argument("--out", default="reports/tables/latency.csv")
    args = parser.parse_args()

    if args.artifacts:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (editable install), for serve/
        from serve.pipeline import ECGPipeline

        pipe = ECGPipeline(args.artifacts, threads=args.threads)
        x = np.load(Path(args.artifacts) / "demo_records.npz")["X"][0] if (Path(args.artifacts) / "demo_records.npz").exists() \
            else np.random.default_rng(0).standard_normal((12, 1000)).astype(np.float32) * 0.2
        meta = json.loads((Path(args.artifacts) / "meta.json").read_text())
        res = {
            "pipeline_end_to_end": time_fn(lambda: pipe.predict(x, force_classify=True), 20, args.n_runs),
            "classifier": benchmark_onnx(Path(args.artifacts) / meta["classifier"]["file"], args.threads, 20, args.n_runs),
            "autoencoder": benchmark_onnx(Path(args.artifacts) / meta["gate"]["file"], args.threads, 20, args.n_runs),
            "threads": args.threads,
            "hardware": hardware(),
        }
        out = Path("reports/tables/serving_latency.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    from ptbxl_ecg.engine import classifier_wrapper, load_checkpoint
    from ptbxl_ecg.models import build_model
    from ptbxl_ecg.utils.config import load_config
    from ptbxl_ecg.models import count_params

    rows = []
    for ck in args.runs:
        if ck.endswith((".yaml", ".yml")):
            cfg = load_config(ck)
            if cfg.get("task") != "classifier":
                continue
            total_len = 10 * cfg["data"]["sampling_rate"]
            module = classifier_wrapper(build_model(cfg["model"]), cfg, total_len).eval()
        else:
            module, ckpt = load_checkpoint(ck, "cpu")
            cfg, total_len = ckpt["config"], ckpt["total_len"]
        stats = benchmark_module(module, args.threads, 20, args.n_runs, total_len)
        rows.append({"name": cfg["name"], "model": cfg["model"]["name"], "n_params": count_params(module),
                     **stats, "threads": args.threads, "hardware": hardware(), "checkpoint": ck})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.out).exists():  # keep rows for other runs, replace re-measured ones
        old = pd.read_csv(args.out)
        df = pd.concat([old[~old["name"].isin(df["name"])], df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
