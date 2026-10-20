import numpy as np
import pytest
import torch

from ptbxl_ecg.data.preprocess import preprocess


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, synthetic_cache):
    """Train tiny models for 1 epoch on synthetic data and export them."""
    from conftest import ROOT
    from ptbxl_ecg.export import export_artifacts
    from ptbxl_ecg.train import run
    from ptbxl_ecg.utils.config import load_config

    runs = tmp_path_factory.mktemp("runs")
    common = [f"data.cache_dir={synthetic_cache}", f"output_dir={runs}", "tracking.backend=none", "train.epochs=1",
              "train.batch_size=32", "train.device=cpu"]
    clf = load_config(ROOT / "configs/c1_resnet1d_bandpass.yaml", common + ["model.widths=[8,8,16,16]", "model.depths=[1,1,1,1]", "model.stem_width=8"])
    ae = load_config(ROOT / "configs/autoencoder.yaml", common + ["model.channels=[8,16,32]"])
    run(clf), run(ae)
    out = tmp_path_factory.mktemp("artifacts")
    export_artifacts(str(runs / clf["name"] / "seed0/best.pt"), str(runs / "autoencoder/seed0/best.pt"), out, n_demo=6)
    return out


@pytest.fixture(scope="module")
def client(artifacts):
    from fastapi.testclient import TestClient

    from serve.app import create_app

    with TestClient(create_app(str(artifacts), threads=1)) as c:
        yield c


def test_serving_preprocessing_matches_training(artifacts):
    """serve/pipeline.py reimplements preprocessing without torch/src; it must agree."""
    from serve.pipeline import ECGPipeline
    from ptbxl_ecg.engine import load_checkpoint

    pipe = ECGPipeline(artifacts, threads=1)
    x = np.load(artifacts / "demo_records.npz")["X"][0]
    wrapper, ckpt = load_checkpoint(pipe.meta["classifier"]["source_checkpoint"])
    with torch.no_grad():
        ref = wrapper(torch.from_numpy(preprocess(x[None], ckpt["norm_mean"], ckpt["norm_std"], ckpt["fs"], True))).numpy()[0]
    got = pipe.classifier(x)
    np.testing.assert_allclose(got, ref, atol=1e-4)


def test_classify_ok_or_low_quality(client, artifacts):
    x = np.load(artifacts / "demo_records.npz")["X"][0]
    r = client.post("/classify", json={"signal": x.tolist()})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "low_quality")
    if body["status"] == "ok":
        assert set(body["probabilities"]) == {"NORM", "MI", "STTC", "CD", "HYP"}
        assert all(0 <= p <= 1 for p in body["probabilities"].values())


def test_garbage_signal_is_refused(client):
    x = np.random.default_rng(0).normal(0, 5.0, (12, 1000))  # 5 mV white noise
    body = client.post("/classify", json={"signal": x.tolist()}).json()
    assert body["status"] == "low_quality"
    assert body["message"] == "signal quality too low"


def test_bad_shape_rejected(client):
    assert client.post("/classify", json={"signal": [[0.0] * 999] * 12}).status_code == 422
    assert client.post("/classify", json={"signal": [[0.0] * 1000] * 11}).status_code == 422


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
