# File 6 — `mlflow_run.py`

**Write:** `learn/src/mlflow_run.py`  
**Reference:** `common/mlflow_run.py`  
**Dependencies:** mlflow (optional — skip if import fails)

---

## Substep 6.1 — Constants and env

`MLFLOW_URI`, `EXPERIMENT`, `MLFLOW_ALLOW_FILE_STORE=true`.

---

## Substep 6.2 — `parse_nvinfer_ini(path)`

`ConfigParser` with `optionxform=str`; return `[property]` section dict.

**Test:** parse `configs/pgie_yolo26.txt` → has `onnx-file` key.

---

## Substep 6.3 — `flatten_params(raw)`

Stringify all values; truncate to 500 chars.

---

## Substep 6.4 — `start_run(run_name, params, experiment=None)`

Set tracking URI, experiment, `start_run`, `log_params`, return `(mlflow, run)`.

---

## Substep 6.5 — `log_finish(mlflow_mod, metrics, artifacts, metrics_path)`

Log numeric metrics, `log_dict` full summary, `log_artifact` for each existing file, `end_run`.

---

## Checkpoint

Call only **after** `log_sink.close()` at EOS. Verify `mlruns/` entry for experiment `tracker_n`.

Next: [07-pipeline_tracker.md](07-pipeline_tracker.md)
