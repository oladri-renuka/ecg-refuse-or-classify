# End-to-end pipeline. Every training target is a config file in configs/.
PY     ?= python
SEEDS  ?= 0 1 2
FRACS  ?= 0.1 0.25 0.5
# Deployed classifier: best *validation* macro AUROC among budget-fitting runs (never test).
# c1_resnet1d_posweight: val 0.9349 (3-seed mean), ~8 ms p95; seed0 has the best val (0.9352).
CLF    ?= runs/c1_resnet1d_posweight/seed0/best.pt
AE     ?= runs/autoencoder/seed0/best.pt
TRAIN   = $(PY) -m ptbxl_ecg.train

.PHONY: setup data data-zip test baseline c3-baseline c1 c2-sweep c2 c3 ae c4c5 c6 gradcam failures latency serve-latency report export serve docker loadtest demo all-gpu

setup:            ## install training env
	pip install -e ".[dev]"

PTBXL_S3      = s3://physionet-open/ptb-xl/1.0.3/
PTBXL_ROOT    = data/raw/ptb-xl-1.0.3
PTBXL_ZIP_URL = https://physionet.org/content/ptb-xl/get-zip/1.0.3/

data:             ## PTB-XL 1.0.3 (CC BY 4.0) from PhysioNet's public S3 mirror, 100 Hz only, then build the cache
	# physionet.org throttles to ~1 MB/min from cloud pods; the S3 mirror is the same files.
	# `sync` is incremental, so rerunning after an interruption only fetches what is missing.
	command -v aws >/dev/null || $(PY) -m pip install -q awscli
	aws s3 sync --no-sign-request --only-show-errors --exclude "records500/*" $(PTBXL_S3) $(PTBXL_ROOT)
	$(PY) -m ptbxl_ecg.data.ptbxl --config configs/base.yaml --root $(PTBXL_ROOT)

data-zip:         ## fallback: single ZIP from physionet.org (slow from cloud IPs), then build the cache
	mkdir -p data/raw
	wget -c --progress=dot:giga -O data/raw/ptb-xl-1.0.3.zip $(PTBXL_ZIP_URL)
	[ -n "$$(find data/raw -name ptbxl_database.csv -print -quit)" ] || $(PY) -m zipfile -e data/raw/ptb-xl-1.0.3.zip data/raw
	$(PY) -m ptbxl_ecg.data.ptbxl --config configs/base.yaml \
	  --root "$$(dirname "$$(find data/raw -name ptbxl_database.csv -print -quit)")"

test:
	$(PY) -m pytest -q

baseline:         ## B0: honest baseline first (CPU is fine)
	$(TRAIN) configs/baseline_logreg.yaml --seeds $(SEEDS)
	$(TRAIN) configs/baseline.yaml --seeds $(SEEDS)

c1:               ## C1: full-size architectures + pos_weight and bandpass ablations
	for m in resnet1d bilstm transformer; do \
	  $(TRAIN) configs/c1_$$m.yaml --seeds $(SEEDS); \
	  $(TRAIN) configs/c1_$${m}_posweight.yaml --seeds $(SEEDS); \
	done
	$(TRAIN) configs/c1_resnet1d_bandpass.yaml --seeds $(SEEDS)

c2-sweep:         ## C2 step 1: latency sweep -> configs/generated/ (run on the LAPTOP CPU)
	$(PY) -m ptbxl_ecg.experiments.latency_sweep configs/c2_sweep.yaml

c2:               ## C2 step 2: train the budget-fitting variants (GPU)
	for f in configs/generated/c2_*.yaml; do $(TRAIN) $$f --seeds $(SEEDS); done

c3:               ## C3: data efficiency (100% is C1)
	for m in resnet1d bilstm transformer; do for f in $(FRACS); do \
	  $(TRAIN) configs/c3_$$m.yaml --seeds $(SEEDS) --set data.train_fraction=$$f; \
	done; done
	$(MAKE) c3-baseline

c3-baseline:      ## C3 control: the B0 baseline on the same 10/25/50% subsets (CPU)
	for f in $(FRACS); do $(TRAIN) configs/c3_baseline.yaml --seeds $(SEEDS) --set data.train_fraction=$$f; done

ae:               ## quality-gate autoencoder
	$(TRAIN) configs/autoencoder.yaml --seeds $(SEEDS)

c4c5:             ## gate AUROC + selective classification
	$(PY) -m ptbxl_ecg.experiments.quality_gate configs/c4c5_quality_gate.yaml

c6:
	$(PY) -m ptbxl_ecg.experiments.robustness configs/c6_robustness.yaml

gradcam:
	$(PY) -m ptbxl_ecg.experiments.gradcam $(CLF) --stage 2

failures:
	$(PY) -m ptbxl_ecg.experiments.failures $(dir $(CLF))

latency:          ## p50/p95 CPU latency + size of every C1/C2 model, from run configs (LAPTOP; no checkpoints needed)
	# trained checkpoints when present (true ONNX size), else the run configs
	$(PY) -m ptbxl_ecg.latency $(or $(wildcard runs/c[12]_*/seed0/best.pt),$(wildcard runs/c[12]_*/seed0/config.yaml))

report:           ## mean ± std tables -> reports/tables + README results block
	$(PY) -m ptbxl_ecg.experiments.aggregate

export:           ## needs the data cache (demo records, ONNX parity check): run on the pod
	$(PY) -m ptbxl_ecg.export --clf $(CLF) --ae $(AE) --out artifacts

serve-latency:    ## end-to-end gate + classifier latency of artifacts/ (run on the LAPTOP)
	$(PY) -m ptbxl_ecg.latency --artifacts artifacts

serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8000

docker:
	docker build -f serve/Dockerfile -t ecg-quality-gate .
	docker run --rm -p 8000:8000 --cpus 4 ecg-quality-gate

loadtest:         ## requires `make serve` or `make docker` running
	locust -f serve/locustfile.py --headless -u 16 -r 4 -t 60s --host http://localhost:8000 --csv reports/tables/locust

demo:
	$(PY) -m serve.stream_demo --artifacts artifacts

all-gpu: baseline c1 c2 c3 ae   ## everything that wants a GPU, in order
