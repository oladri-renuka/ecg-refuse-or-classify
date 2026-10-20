from conftest import ROOT

from ptbxl_ecg.utils.config import load_config


def test_inheritance_and_overrides():
    cfg = load_config(ROOT / "configs/c1_transformer_posweight.yaml", ["train.epochs=3", "data.train_folds=[1,2]"])
    assert cfg["model"]["name"] == "transformer"
    assert cfg["train"]["lr"] == 3e-4            # from c1_transformer
    assert cfg["train"]["pos_weight"] is True    # from the posweight child
    assert cfg["train"]["epochs"] == 3           # CLI override
    assert cfg["data"]["train_folds"] == [1, 2]
    assert cfg["data"]["val_folds"] == [9]       # from base


def test_name_placeholders():
    cfg = load_config(ROOT / "configs/c3_bilstm.yaml", ["data.train_fraction=0.25"])
    assert cfg["name"] == "c3_bilstm_frac0.25"
    assert cfg["experiment"] == "C3"


def test_all_configs_load():
    for path in (ROOT / "configs").glob("*.yaml"):
        cfg = load_config(path)
        assert isinstance(cfg, dict)


def test_base_yaml_agrees_with_package_defaults():
    """Downstream projects use DEFAULT_DATA without any YAML; it must equal base.yaml."""
    from ptbxl_ecg.data import DEFAULT_DATA

    data = load_config(ROOT / "configs/base.yaml")["data"]
    assert {k: data[k] for k in DEFAULT_DATA} == DEFAULT_DATA
