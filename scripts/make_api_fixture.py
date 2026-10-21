"""Write the two requests the Docker CI smoke test sends to POST /classify.

    python scripts/make_api_fixture.py

  tests/fixtures/classify_ok.json           a clean PTB-XL fold-10 record   -> expect "ok"
  tests/fixtures/classify_low_quality.json  the same record + 0.3 mV noise  -> expect "low_quality"

Both are checked against the in-process pipeline before writing, so the fixture can
never silently disagree with the exported models it is meant to test.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.pipeline import ECGPipeline  # noqa: E402

pipe = ECGPipeline(ROOT / "artifacts", threads=1)
demo = np.load(ROOT / "artifacts" / "demo_records.npz")
clean = next(i for i in range(len(demo["X"])) if not demo["noisy"][i] and pipe.predict(demo["X"][i])["status"] == "ok")
x = np.round(demo["X"][clean].astype(np.float64), 4)  # 4 decimals of mV keeps the file small
noisy = np.round(x + np.random.default_rng(0).normal(0, 0.3, x.shape), 4)

out = ROOT / "tests" / "fixtures"
out.mkdir(parents=True, exist_ok=True)
for name, sig, expect in (("classify_ok", x, "ok"), ("classify_low_quality", noisy, "low_quality")):
    got = pipe.predict(sig)
    assert got["status"] == expect, (name, got)
    (out / f"{name}.json").write_text(json.dumps({"signal": sig.tolist()}, separators=(",", ":")))
    print(f"{name}.json  ecg_id={int(demo['ecg_id'][clean])}  status={got['status']}  "
          f"error={got['reconstruction_error']:.3f} (threshold {got['quality_threshold']:.3f})  labels={got.get('labels')}")
