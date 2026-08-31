#!/usr/bin/env python3
"""Print FPS / latency comparison for probe vs async MLflow runs."""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:/workspace/mlruns")

KEYS = (
    "fps_e2e",
    "latency_e2e_ms_avg",
    "probe_log_us_avg",
    "dropped_frames",
    "frames_osd",
    "gpu_util_avg",
)


def latest_by_mode(client, exp_id: str) -> dict:
    found = {}
    runs = client.search_runs([exp_id], order_by=["attributes.start_time DESC"], max_results=50)
    for run in runs:
        mode = run.data.params.get("log_mode")
        if mode in ("probe", "async") and mode not in found:
            found[mode] = run
        if len(found) == 2:
            break
    return found


def print_table(found: dict) -> int:
    if "probe" not in found or "async" not in found:
        print("Need at least one probe run and one async run in experiment 'detector'.")
        print("Have:", sorted(found))
        return 1
    print(f"{'metric':<24} {'probe':>14} {'async':>14} {'delta(async-probe)':>20}")
    print("-" * 74)
    for key in KEYS:
        p = found["probe"].data.metrics.get(key)
        a = found["async"].data.metrics.get(key)
        if p is None or a is None:
            print(f"{key:<24} {str(p):>14} {str(a):>14}")
            continue
        print(f"{key:<24} {p:14.4f} {a:14.4f} {a - p:20.4f}")
    print()
    print("probe run:", found["probe"].info.run_name, found["probe"].info.run_id[:8])
    print("async run:", found["async"].info.run_name, found["async"].info.run_id[:8])
    return 0


def attach_nsys(path: str) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("file:/workspace/mlruns")
    client = MlflowClient()
    exp = client.get_experiment_by_name("detector")
    if exp is None:
        return
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return
    mlflow.set_experiment("detector")
    with mlflow.start_run(run_id=runs[0].info.run_id):
        mlflow.log_artifact(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attach-nsys", default=None)
    args = parser.parse_args()
    if args.attach_nsys:
        attach_nsys(args.attach_nsys)
        return 0

    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri("file:/workspace/mlruns")
    client = MlflowClient()
    exp = client.get_experiment_by_name("detector")
    if exp is None:
        print("No MLflow experiment 'detector' yet. Run ./run.sh --log-mode probe and async first.")
        return 1
    return print_table(latest_by_mode(client, exp.experiment_id))


if __name__ == "__main__":
    sys.exit(main())
