# Serving-only image: FastAPI + ONNX Runtime + exported models (no PyTorch, no training code).
# Build from the repo root after `make export`. On Apple Silicon, target x86 servers explicitly:
#   docker build --platform linux/amd64 -f serve/Dockerfile -t ecg-refuse-or-classify .
#   docker run --rm -p 8000:8000 --cpus 4 ecg-refuse-or-classify
# PORT is configurable (Hugging Face Docker Spaces route to the port named in the Space card).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ECG_ARTIFACTS=/app/artifacts ECG_THREADS=4 OMP_NUM_THREADS=4 PORT=8000

WORKDIR /app
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt

COPY serve/__init__.py serve/pipeline.py serve/app.py serve/
# meta.json carries preprocessing stats, per-class thresholds and the quality-gate threshold
COPY artifacts/meta.json artifacts/classifier.onnx artifacts/autoencoder.onnx artifacts/

# uid 1000 matches what Hugging Face Spaces runs containers as
RUN useradd --create-home --uid 1000 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "exec uvicorn serve.app:app --host 0.0.0.0 --port ${PORT} --workers 1"]
