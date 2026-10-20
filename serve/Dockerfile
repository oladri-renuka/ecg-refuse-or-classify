# Build from the repo root after `python -m ptbxl_ecg.export ... --out artifacts`:
#   docker build -f serve/Dockerfile -t ecg-quality-gate .
#   docker run --rm -p 8000:8000 --cpus 4 ecg-quality-gate
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    ECG_ARTIFACTS=/app/artifacts ECG_THREADS=4 OMP_NUM_THREADS=4

WORKDIR /app
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --no-cache-dir -r serve/requirements.txt

COPY serve/__init__.py serve/pipeline.py serve/app.py serve/
COPY artifacts/meta.json artifacts/classifier.onnx artifacts/autoencoder.onnx artifacts/

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
