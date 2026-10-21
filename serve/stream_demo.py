"""Streaming demo: replay a PTB-XL fold-10 ECG in 1-second chunks, classify the rolling
10-second window every second, and show the decision live.

    python -m serve.stream_demo                                  # in-process pipeline
    python -m serve.stream_demo --api http://localhost:8000      # call the running API
"""
from __future__ import annotations

import argparse
import html
import time
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from serve.pipeline import ECGPipeline

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SHOW_LEADS = ["I", "II", "V2", "V5"]
CLASS_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}
INK = "#1f3b57"
POSITIVE = "#0f8a5f"
NEGATIVE = "#b8c2cc"

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#hero h1 {font-size: 1.7rem; margin-bottom: 0.2rem;}
#hero p {opacity: 0.8; margin-top: 0;}
.card {border-radius: 14px; padding: 14px 18px; border: 1px solid var(--border-color-primary);
       background: var(--block-background-fill);}
.banner {border-radius: 14px; padding: 16px 20px; display: flex; align-items: center; gap: 14px;}
.banner .icon {font-size: 2rem; line-height: 1;}
.banner .title {font-size: 1.25rem; font-weight: 700;}
.banner .sub {opacity: 0.85; font-size: 0.92rem;}
.banner.ok {background: #e7f6ee; border: 1px solid #9ad7b8;}
.banner.ok, .banner.ok * {color: #0b5c3f !important;}
.banner.refused {background: #fdecec; border: 1px solid #f2a7a7;}
.banner.refused, .banner.refused * {color: #8a1f1f !important;}
.banner.wait {background: #eef1f5; border: 1px solid #cfd7e0;}
.banner.wait, .banner.wait * {color: #3b4a5a !important;}
.banner .chips span.pos {color: white !important;}
.chips span {display: inline-block; padding: 3px 10px; margin: 3px 4px 0 0; border-radius: 999px;
             font-size: 0.85rem; font-weight: 600; background: #e6eef7; color: #1f3b57;}
.chips span.pos {background: #0f8a5f; color: white;}
.meter {height: 12px; border-radius: 999px; background: #e9edf2; position: relative; overflow: visible; margin: 8px 0 4px;}
.meter .fill {height: 100%; border-radius: 999px;}
.meter .tick {position: absolute; top: -5px; width: 2px; height: 22px; background: #333;}
.kv {display: grid; grid-template-columns: auto 1fr; gap: 4px 14px; font-size: 0.92rem;}
.kv b {opacity: 0.7; font-weight: 600;}
.muted {opacity: 0.7; font-size: 0.85rem;}
footer {display: none !important;}
"""

# ECG paper and the charts are light by nature; keep the whole page in the light theme
# so they don't sit as white boxes on a dark page when the OS is in dark mode.
HEAD = """<script>
(() => { const u = new URL(window.location.href);
  if (u.searchParams.get("__theme") !== "light") { u.searchParams.set("__theme", "light"); window.location.replace(u); } })();
</script>"""


def ecg_figure(window: np.ndarray, fs: int, total: int) -> plt.Figure:
    """Classic ECG paper: 0.2 s / 0.5 mV major grid, 0.04 s / 0.1 mV minor grid."""
    n = len(SHOW_LEADS)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.55 * n), sharex=True)
    fig.patch.set_facecolor("white")
    t = np.arange(window.shape[1]) / fs
    for ax, name in zip(axes, SHOW_LEADS):
        y = window[LEADS.index(name)]
        ax.set_facecolor("#fff6f6")
        ax.set_xlim(0, total / fs)
        lo, hi = (float(y.min()), float(y.max())) if len(y) else (-1.0, 1.0)
        pad = max(0.3, 0.1 * (hi - lo))
        ax.set_ylim(min(lo - pad, -0.6), max(hi + pad, 0.6))
        ax.set_xticks(np.arange(0, total / fs + 1e-9, 0.2), minor=False)
        ax.set_xticks(np.arange(0, total / fs + 1e-9, 0.04), minor=True)
        ax.set_yticks(np.arange(np.floor(ax.get_ylim()[0] * 2) / 2, ax.get_ylim()[1], 0.5))
        ax.set_yticks(np.arange(np.floor(ax.get_ylim()[0] * 10) / 10, ax.get_ylim()[1], 0.1), minor=True)
        ax.grid(which="major", color="#f0a8a8", linewidth=0.6)
        ax.grid(which="minor", color="#fbd9d9", linewidth=0.3)
        ax.tick_params(which="both", length=0, labelleft=False, labelbottom=False)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.plot(t, y, color="#111", linewidth=0.9)
        ax.text(0.006, 0.86, name, transform=ax.transAxes, fontsize=10, fontweight="bold", color="#7a1c1c")
    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xticks(np.arange(0, total / fs + 1e-9, 1.0))
    axes[-1].set_xticklabels([f"{s:g}s" for s in np.arange(0, total / fs + 1e-9, 1.0)], fontsize=8, color="#555")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.08, hspace=0.04)
    plt.close(fig)
    return fig


def prob_figure(result: dict | None, classes: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    fig.patch.set_facecolor("white")
    probs = [result["probabilities"][c] for c in classes] if result and result.get("status") == "ok" else [0.0] * len(classes)
    thr = [result["thresholds"][c] for c in classes] if result and result.get("status") == "ok" else [None] * len(classes)
    ypos = np.arange(len(classes))[::-1]
    for y, c, p, t in zip(ypos, classes, probs, thr):
        ax.barh(y, 1.0, color="#eef1f5", height=0.62)
        ax.barh(y, p, color=POSITIVE if (t is not None and p >= t) else NEGATIVE, height=0.62)
        if t is not None:
            ax.plot([t, t], [y - 0.38, y + 0.38], color="#222", linewidth=1.6)
        ax.text(1.02, y, f"{p:.2f}" if t is not None else "–", va="center", fontsize=9, color="#333")
    ax.set_yticks(ypos, [f"{c}  {CLASS_NAMES[c]}" for c in classes], fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"], fontsize=8, color="#666")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("probability  ( | = frozen decision threshold )", fontsize=8.5, color="#666", loc="left")
    fig.tight_layout()
    plt.close(fig)
    return fig


def banner(kind: str, title: str, sub: str) -> str:
    icon = {"ok": "✅", "refused": "⛔", "wait": "⏳"}[kind]
    return f'<div class="banner {kind}"><div class="icon">{icon}</div><div><div class="title">{html.escape(title)}</div><div class="sub">{sub}</div></div></div>'


def gate_card(result: dict | None) -> str:
    if not result or "reconstruction_error" not in result:
        return '<div class="card"><b>Signal-quality gate</b><div class="muted">waiting for a full 10 s window…</div></div>'
    err, thr = result["reconstruction_error"], result["quality_threshold"]
    scale = max(thr * 3, err * 1.05)
    fill = "#0f8a5f" if err <= thr else "#c0392b"
    verdict = "passes" if err <= thr else "rejected"
    return (
        '<div class="card"><b>Signal-quality gate</b>'
        f'<div class="meter"><div class="fill" style="width:{min(100, 100 * err / scale):.1f}%;background:{fill}"></div>'
        f'<div class="tick" style="left:{100 * thr / scale:.1f}%"></div></div>'
        f'<div class="muted">autoencoder reconstruction error <b>{err:.3f}</b> vs threshold {thr:.3f}: <b>{verdict}</b></div></div>'
    )


def record_card(ecg_id: int, true_labels: list[str], noisy: bool) -> str:
    chips = "".join(f"<span>{c}</span>" for c in true_labels) or "<span>none</span>"
    return (
        '<div class="card"><div class="kv">'
        f"<b>Record</b><span>PTB-XL ecg_id {ecg_id} (test fold 10)</span>"
        f'<b>Cardiologist labels</b><span class="chips">{chips}</span>'
        f"<b>Metadata noise flag</b><span>{'yes' if noisy else 'no'}</span>"
        "</div></div>"
    )


def build(artifacts: str, api: str | None) -> gr.Blocks:
    demo_data = np.load(Path(artifacts) / "demo_records.npz", allow_pickle=False)
    X, Y, ecg_ids, noisy = demo_data["X"], demo_data["Y"], demo_data["ecg_id"], demo_data["noisy"]
    classes = [str(c) for c in demo_data["classes"]]
    pipe = ECGPipeline(artifacts) if api is None else None
    fs = int(pipe.meta["fs"]) if pipe else 100
    win = 10 * fs

    def classify(window: np.ndarray) -> dict:
        if pipe is not None:
            t0 = time.perf_counter()
            out = pipe.predict(window)
            out["latency_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
            return out
        import httpx

        return httpx.post(f"{api}/classify", json={"signal": window.tolist()}, timeout=10).json()

    def labels_of(i: int) -> list[str]:
        return [c for c, y in zip(classes, Y[i]) if y]

    def describe(choice: str) -> str:
        i = int(choice.split(":")[0])
        return record_card(int(ecg_ids[i]), labels_of(i), bool(noisy[i]))

    def stream(choice: str, noise_mv: float, wander_mv: float, seconds: int):
        i = int(choice.split(":")[0])
        record = X[i]
        rng = np.random.default_rng(0)
        f, phase = rng.uniform(0.15, 0.35), rng.uniform(0, 2 * np.pi)
        buffer = np.zeros((12, 0), dtype=np.float32)
        result = None
        for sec in range(int(seconds)):
            t_start = time.perf_counter()
            start = (sec * fs) % win
            chunk = record[:, start : start + fs].copy()
            if noise_mv > 0:
                chunk += rng.normal(0, noise_mv, chunk.shape).astype(np.float32)
            if wander_mv > 0:
                t = (sec * fs + np.arange(fs)) / fs
                chunk += (wander_mv * np.sin(2 * np.pi * f * t + phase)).astype(np.float32)
            buffer = np.concatenate([buffer, chunk], axis=1)[:, -win:]

            if buffer.shape[1] < win:
                result = None
                head = banner("wait", f"Buffering {buffer.shape[1] // fs} / 10 s", "the classifier needs a full 10-second window")
            else:
                result = classify(buffer)
                if result["status"] == "ok":
                    found = result["labels"]
                    chips = "".join(f'<span class="pos">{c}</span>' for c in found) or "<span>no class above threshold</span>"
                    head = banner("ok", "Classified", f'<span class="chips">{chips}</span>&nbsp; {result.get("latency_ms", "–")} ms · window ending at {sec + 1} s')
                else:
                    head = banner("refused", "Signal quality too low: not classified",
                                  "the quality gate refused this window instead of guessing")
            yield head, gate_card(result), ecg_figure(buffer, fs, win), prob_figure(result, classes), result or {"status": "buffering"}
            time.sleep(max(0.0, 1.0 - (time.perf_counter() - t_start)))

    choices = [
        f"{i}: ecg_id {e}  ·  {'|'.join(labels_of(i)) or '-'}{'  ·  metadata: noisy' if n else ''}"
        for i, (e, n) in enumerate(zip(ecg_ids, noisy))
    ]
    with gr.Blocks(title="ECG quality-gate demo") as ui:
        gr.Markdown(
            "# Real-time 12-lead ECG classification with a quality gate\n"
            "A PTB-XL test ECG is replayed in 1-second chunks; every second the latest 10 s are checked by the "
            "autoencoder gate and, if they pass, classified into 5 diagnostic superclasses. "
            "Corrupted signals are **refused instead of guessed**. Add noise or baseline wander to see the gate act.",
            elem_id="hero",
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=300):
                rec = gr.Dropdown(choices, value=choices[0], label="Test record")
                info = gr.HTML(describe(choices[0]))
                noise = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="Inject Gaussian noise (mV)")
                wander = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Inject baseline wander (mV)")
                secs = gr.Slider(10, 60, value=20, step=1, label="Stream length (s)")
                with gr.Row():
                    go = gr.Button("▶  Start stream", variant="primary")
                    stop = gr.Button("■  Stop", variant="secondary")
            with gr.Column(scale=8):
                status = gr.HTML(banner("wait", "Ready", "press Start stream"))
                gate = gr.HTML(gate_card(None))
                probs = gr.Plot(prob_figure(None, classes), label="Class probabilities", show_label=False)
        gr.Markdown("**Rolling 10-second window**: leads I, II, V2, V5 on standard ECG paper "
                    "(large squares 0.2 s × 0.5 mV)")
        ecg = gr.Plot(ecg_figure(np.zeros((12, 0)), fs, win), show_label=False)
        with gr.Accordion("Raw API response", open=False):
            raw = gr.JSON(label=None)
        gr.Markdown(
            f"<div class='muted'>Classifier: <code>{pipe.meta['classifier']['config_name'] if pipe else 'remote API'}</code> "
            "(ONNX Runtime, CPU) · thresholds frozen on validation fold 9 · research prototype trained on PTB-XL, "
            "<b>not a medical device</b>.</div>"
        )
        rec.change(describe, rec, info)
        run = go.click(stream, inputs=[rec, noise, wander, secs], outputs=[status, gate, ecg, probs, raw])
        stop.click(None, cancels=[run])
    return ui


THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate", font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--api", default=None)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build(args.artifacts, args.api).launch(server_port=args.port, theme=THEME, css=CSS, head=HEAD)


if __name__ == "__main__":
    main()
