#!/usr/bin/env bash
# Everything that needs PTB-XL runs on the pod; the data never has to touch the laptop.
#
#   laptop:  make c2-sweep && git commit configs/generated && git push   (needs no data)
#   pod:     bash scripts/runpod.sh check     # download + cache + baseline + 1 short run (~foreground)
#            bash scripts/runpod.sh train     # all GPU runs, 3 seeds, in the background
#            bash scripts/runpod.sh analyze   # C4-C6, Grad-CAM, failures, export, report
#   laptop:  pull runs/ reports/ artifacts/ back, then `make latency serve-latency report`
set -euo pipefail
stage=${1:-check}

# Install into a venv that still sees the image's preinstalled torch. Installing into the
# system Python fails on RunPod images: pip cannot uninstall Debian-managed packages
# (blinker, cryptography, ...). Every stage activates the same venv.
VENV=${VENV:-/workspace/venv}
if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  python3 -m venv --system-site-packages "$VENV" 2>/dev/null || {
    apt-get update -qq && apt-get install -y -qq python3-venv
    rm -rf "$VENV" && python3 -m venv --system-site-packages "$VENV"
  }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

case "$stage" in
  check)
    pip install -q --upgrade pip
    pip install -q -e ".[dev]"
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    python -m pytest -q
    [ -f data/processed/X_100.npy ] || make data
    echo "---- cache summary: expect ~21.4k kept of 21,799, NORM ~9.5k, HYP ~2.6k ----"
    cat data/processed/info.json
    ls configs/generated/c2_*.yaml >/dev/null 2>&1 \
      || { echo "configs/generated/ is empty: run 'make c2-sweep' on the laptop, commit, pull"; exit 1; }
    make baseline SEEDS=0
    # 5-epoch sanity run: val AUROC should already be well above 0.8
    python -m ptbxl_ecg.train configs/c1_resnet1d.yaml --seeds 0 --set train.epochs=5 name=sanity_resnet1d
    echo "check passed: look at the epoch times above to size the full run"
    ;;
  train)
    rm -rf runs/sanity_resnet1d
    nohup make all-gpu > runs_gpu.log 2>&1 &
    echo "training in background: tail -f runs_gpu.log"
    echo "(the venv is activated inside this script; for manual commands run: source $VENV/bin/activate)"
    ;;
  analyze)
    make c3-baseline c4c5 c6 gradcam failures export report
    echo "now copy runs/ reports/ artifacts/ to the laptop (see README > Compute)"
    ;;
  *) echo "usage: $0 {check|train|analyze}"; exit 2 ;;
esac
