"""Thin wrapper so training code doesn't care whether MLflow, W&B or nothing is used.

Local JSON files in the run directory remain the source of truth for the results
tables; the tracker is for dashboards and run comparison.
"""
from __future__ import annotations

from typing import Any

from ptbxl_ecg.utils.config import flatten


class Tracker:
    def __init__(self, cfg: dict, run_name: str):
        tcfg = cfg.get("tracking", {})
        self.backend = tcfg.get("backend", "none")
        self._run = None
        params = {k: v for k, v in flatten(cfg).items() if not k.startswith("_")}
        if self.backend == "mlflow":
            import mlflow

            if tcfg.get("tracking_uri"):
                mlflow.set_tracking_uri(tcfg["tracking_uri"])
            mlflow.set_experiment(tcfg.get("project", "default"))
            self._run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.set_tags({"experiment_id": cfg.get("experiment", ""), "config": cfg.get("_config_path", "")})
            self._mlflow = mlflow
        elif self.backend == "wandb":
            import wandb

            self._run = wandb.init(
                project=tcfg.get("project", "default"),
                name=run_name,
                group=cfg.get("name"),
                tags=[str(cfg.get("experiment", ""))],
                config=params,
                reinit=True,
            )
        elif self.backend != "none":
            raise ValueError(f"Unknown tracking backend {self.backend!r}")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        scalars = {k: float(v) for k, v in flatten(metrics).items() if isinstance(v, (int, float))}
        if self.backend == "mlflow":
            self._mlflow.log_metrics(scalars, step=step)
        elif self.backend == "wandb":
            self._run.log(scalars, step=step)

    def log_artifact(self, path: str) -> None:
        if self.backend == "mlflow":
            self._mlflow.log_artifact(path)
        elif self.backend == "wandb":
            self._run.save(path, policy="now")

    def finish(self) -> None:
        if self.backend == "mlflow":
            self._mlflow.end_run()
        elif self.backend == "wandb":
            self._run.finish()
