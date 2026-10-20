from __future__ import annotations

from torch import nn

from ptbxl_ecg.models.autoencoder import ConvAutoencoder
from ptbxl_ecg.models.lstm import BiLSTM
from ptbxl_ecg.models.resnet1d import ResNet1d
from ptbxl_ecg.models.transformer import ECGTransformer

REGISTRY = {
    "resnet1d": ResNet1d,
    "bilstm": BiLSTM,
    "transformer": ECGTransformer,
    "autoencoder": ConvAutoencoder,
}


def build_model(mcfg: dict) -> nn.Module:
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in mcfg.items() if k != "name"}
    return REGISTRY[mcfg["name"]](**kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
