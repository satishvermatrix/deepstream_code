"""Local-file MLflow helpers. Call after the async logger has joined (EOS)."""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

MLFLOW_URI = "file:/workspace/mlruns"
EXPERIMENT = "detector"


def parse_nvinfer_ini(path: str) -> dict[str, str]:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(path)
    if "property" not in cfg:
        return {}
    return {k: str(v) for k, v in cfg["property"].items()}


def flatten_params(raw: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in raw.items():
        s = str(val)
        out[key] = s[:500]
    return out


def start_run(run_name: str | None, params: dict[str, Any], experiment: str | None = None):
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment or EXPERIMENT)
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(flatten_params(params))
    return mlflow, run


def log_finish(
    mlflow_mod,
    metrics: dict[str, Any],
    artifacts: list[str],
    metrics_path: str,
) -> None:
    numeric = {}
    for key, val in metrics.items():
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            numeric[key] = float(val)
    mlflow_mod.log_metrics(numeric)
    mlflow_mod.log_dict(metrics, Path(metrics_path).name)
    for path in artifacts:
        p = Path(path)
        if p.is_file():
            mlflow_mod.log_artifact(str(p))
    mlflow_mod.end_run()
