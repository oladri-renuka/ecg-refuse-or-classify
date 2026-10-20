import pytest
import torch

from conftest import ROOT
from ptbxl_ecg.models import build_model, count_params
from ptbxl_ecg.models.wrappers import SlidingWindowClassifier, window_starts
from ptbxl_ecg.utils.config import load_config


@pytest.mark.parametrize("name,lo,hi", [("c1_resnet1d", 2e6, 8e6), ("c1_bilstm", 1.8e6, 3e6), ("c1_transformer", 0.6e6, 1.4e6)])
def test_classifier_shapes_and_size(name, lo, hi):
    model = build_model(load_config(ROOT / f"configs/{name}.yaml")["model"]).eval()
    assert lo <= count_params(model) <= hi
    for T in (250, 1000):
        assert model(torch.randn(3, 12, T)).shape == (3, 5)


def test_autoencoder_shape_and_bottleneck():
    ae = build_model(load_config(ROOT / "configs/autoencoder.yaml")["model"]).eval()
    x = torch.randn(2, 12, 1000)
    assert ae.encoder(x).shape == (2, 128, 125)
    assert ae(x).shape == x.shape
    assert 0.2e6 <= count_params(ae) <= 0.7e6


def test_window_starts_cover_record():
    assert window_starts(1000, 250, 125) == [0, 125, 250, 375, 500, 625, 750]
    assert window_starts(1000, 300, 250)[-1] == 700


def test_sliding_window_equals_manual_average():
    torch.manual_seed(0)
    model = build_model({"name": "resnet1d", "widths": [8, 8, 8, 8], "depths": [1, 1, 1, 1], "stem_width": 8}).eval()
    wrap = SlidingWindowClassifier(model, 250, 125, 1000).eval()
    x = torch.randn(3, 12, 1000)
    with torch.no_grad():
        manual = torch.stack([torch.sigmoid(model(x[:, :, s : s + 250])) for s in wrap.starts]).mean(0)
        torch.testing.assert_close(wrap(x), manual)
