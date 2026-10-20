"""PTB-XL ECG classification with a signal-quality gate.

Public API (stable across minor versions; everything else is internal):
    load_cache, make_splits, Splits    - the cached PTB-XL arrays and the benchmark split
    compute_norm_stats, preprocess     - train-fold standardization (+ optional bandpass)
    build_model, load_checkpoint       - model zoo and checkpoint loading
    Evaluator                          - frozen classifier for scoring raw mV records
    SUPERCLASSES, LEADS, NOISE_COLS, DEFAULT_DATA, PIPELINE_VERSION
"""
__version__ = "0.1.0"

from ptbxl_ecg.data import DEFAULT_DATA, LEADS, NOISE_COLS, PIPELINE_VERSION, SUPERCLASSES
from ptbxl_ecg.data.dataset import PipelineVersionError, Splits, load_cache, make_splits
from ptbxl_ecg.data.preprocess import compute_norm_stats, preprocess, standardize
from ptbxl_ecg.engine import load_checkpoint
from ptbxl_ecg.evaluator import Evaluator
from ptbxl_ecg.models import build_model

__all__ = [
    "__version__", "DEFAULT_DATA", "LEADS", "NOISE_COLS", "PIPELINE_VERSION", "SUPERCLASSES",
    "PipelineVersionError", "Splits", "load_cache", "make_splits", "compute_norm_stats", "preprocess",
    "standardize", "load_checkpoint", "Evaluator", "build_model",
]
