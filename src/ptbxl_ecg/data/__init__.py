SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
NOISE_COLS = ["static_noise", "burst_noise", "baseline_drift", "electrodes_problems"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Bump whenever the cache format, label aggregation, or preprocessing changes in a
# way that alters numbers. Written into the cache's info.json and every checkpoint;
# loading a mismatched cache or checkpoint raises instead of silently drifting.
PIPELINE_VERSION = 1

# The benchmark protocol. configs/base.yaml must agree (enforced by tests/test_config.py),
# so downstream code that never sees a YAML file gets the identical split.
DEFAULT_DATA = {
    "sampling_rate": 100,
    "bandpass": False,
    "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
    "val_folds": [9],
    "test_folds": [10],
    "train_fraction": 1.0,
    "subset_seed": 0,
}
