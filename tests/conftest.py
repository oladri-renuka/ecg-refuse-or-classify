import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # serve/ lives at the repo root, outside the package


@pytest.fixture(scope="session")
def synthetic_cache(tmp_path_factory):
    from ptbxl_ecg.data.ptbxl import make_synthetic

    out = tmp_path_factory.mktemp("cache")
    make_synthetic(out, n=240, seed=0)
    return out


@pytest.fixture
def tiny_cfg(synthetic_cache, tmp_path):
    """c1_resnet1d shrunk to run in seconds on CPU against the synthetic cache."""
    from ptbxl_ecg.utils.config import load_config

    return load_config(
        ROOT / "configs/c1_resnet1d.yaml",
        [
            f"data.cache_dir={synthetic_cache}",
            f"output_dir={tmp_path / 'runs'}",
            "tracking.backend=none",
            "train.epochs=2",
            "train.batch_size=32",
            "train.device=cpu",
            "model.widths=[8,16,16,32]",
            "model.depths=[1,1,1,1]",
            "model.stem_width=8",
        ],
    )
