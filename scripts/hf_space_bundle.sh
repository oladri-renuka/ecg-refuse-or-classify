#!/usr/bin/env bash
# Assemble a Hugging Face Docker Space from the serving image's inputs.
#   bash scripts/hf_space_bundle.sh            # -> dist/hf-space/
# Then push dist/hf-space/ to a new Space (sdk: docker). Files >10 MB need Git LFS:
#   cd dist/hf-space && git init && git lfs install && git lfs track "*.onnx" \
#     && git add . && git commit -m "ECG quality-gate API" \
#     && git remote add origin https://huggingface.co/spaces/<user>/<space> && git push -u origin main
set -euo pipefail
OUT=dist/hf-space
rm -rf "$OUT" && mkdir -p "$OUT/serve" "$OUT/artifacts"
cp serve/Dockerfile "$OUT/Dockerfile"                       # COPY paths are relative to the Space root
cp serve/__init__.py serve/pipeline.py serve/app.py serve/requirements.txt "$OUT/serve/"
cp artifacts/meta.json artifacts/classifier.onnx artifacts/autoencoder.onnx "$OUT/artifacts/"
cp deploy/huggingface/README.md "$OUT/README.md"            # Space card: sdk docker, app_port 8000
# free CPU Spaces have 2 vCPUs
sed -i.bak 's/ECG_THREADS=4 OMP_NUM_THREADS=4/ECG_THREADS=2 OMP_NUM_THREADS=2/' "$OUT/Dockerfile" && rm "$OUT/Dockerfile.bak"
du -sh "$OUT" && find "$OUT" -type f | sort
