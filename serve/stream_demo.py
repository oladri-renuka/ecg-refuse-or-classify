"""Streaming demo: replay a fold-10 ECG in 1-second chunks, classify the rolling
10-second window every second, show it live.

    python -m serve.stream_demo                                  # in-process pipeline
    python -m serve.stream_demo --api http://localhost:8000      # call the running API
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from serve.pipeline import ECGPipeline

LEADS_TO_PLOT = [1, 7]  # II, V2


def build(artifacts: str, api: str | None) -> gr.Blocks:
    demo_data = np.load(Path(artifacts) / "demo_records.npz", allow_pickle=False)
    X, ecg_ids, noisy = demo_data["X"], demo_data["ecg_id"], demo_data["noisy"]
    pipe = ECGPipeline(artifacts) if api is None else None
    fs = int(pipe.meta["fs"]) if pipe else 100
    win = 10 * fs

    def classify(window: np.ndarray) -> dict:
        if pipe is not None:
            return pipe.predict(window)
        import httpx

        return httpx.post(f"{api}/classify", json={"signal": window.tolist()}, timeout=10).json()

    def stream(choice: str, noise_mv: float, seconds: int):
        i = int(choice.split(":")[0])
        record = X[i]
        rng = np.random.default_rng(0)
        buffer = np.zeros((12, 0), dtype=np.float32)
        for sec in range(int(seconds)):
            chunk = record[:, (sec * fs) % win : (sec * fs) % win + fs]
            if noise_mv > 0:
                chunk = chunk + rng.normal(0, noise_mv, chunk.shape).astype(np.float32)
            buffer = np.concatenate([buffer, chunk], axis=1)[:, -win:]
            fig, axes = plt.subplots(len(LEADS_TO_PLOT), 1, figsize=(10, 3.2), sharex=True)
            for ax, lead in zip(axes, LEADS_TO_PLOT):
                ax.plot(np.arange(buffer.shape[1]) / fs, buffer[lead], lw=0.8, color="#1f4e79")
                ax.set_ylabel(["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"][lead])
                ax.set_xlim(0, 10)
            axes[-1].set_xlabel("seconds in rolling window")
            fig.tight_layout()
            if buffer.shape[1] < win:
                result = {"status": "buffering", "seconds_received": sec + 1}
            else:
                result = classify(buffer)
            plt.close(fig)
            yield fig, result
            time.sleep(1.0)

    choices = [f"{i}: ecg_id {e}{' (metadata: noisy)' if n else ''}" for i, (e, n) in enumerate(zip(ecg_ids, noisy))]
    with gr.Blocks(title="Streaming ECG classification") as ui:
        gr.Markdown("### Rolling 10 s window, classified every second. Signals failing the quality gate are refused, not guessed.")
        with gr.Row():
            rec = gr.Dropdown(choices, value=choices[0], label="Test record (PTB-XL fold 10)")
            noise = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="Inject Gaussian noise (mV)")
            secs = gr.Slider(10, 60, value=20, step=1, label="Stream length (s)")
        go = gr.Button("Start stream")
        plot = gr.Plot()
        out = gr.JSON(label="API response")
        go.click(stream, inputs=[rec, noise, secs], outputs=[plot, out])
    return ui


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--api", default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build(args.artifacts, args.api).launch(server_port=args.port)


if __name__ == "__main__":
    main()
