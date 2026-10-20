"""Load test.

    locust -f serve/locustfile.py --headless -u 16 -r 4 -t 60s --host http://localhost:8000 --csv reports/tables/locust
"""
from pathlib import Path

import numpy as np
from locust import HttpUser, between, task

_DEMO = Path(__file__).resolve().parents[1] / "artifacts" / "demo_records.npz"
RECORDS = np.load(_DEMO)["X"] if _DEMO.exists() else np.random.default_rng(0).normal(0, 0.2, (8, 12, 1000)).astype(np.float32)
PAYLOADS = [{"signal": r.tolist()} for r in RECORDS]


class ECGClient(HttpUser):
    wait_time = between(0.0, 0.01)

    @task
    def classify(self):
        payload = PAYLOADS[np.random.randint(len(PAYLOADS))]
        with self.client.post("/classify", json=payload, catch_response=True) as resp:
            if resp.status_code != 200 or resp.json().get("status") not in ("ok", "low_quality"):
                resp.failure(f"bad response {resp.status_code}")
