# Step 0 — Prerequisites

No code file yet. Set up the environment and tutorial tree.

## Substep 0.1 — Container and assets

```sh
chmod +x dev.sh setup_tracker.sh run_tracker_multisrc.sh
./dev.sh up
./dev.sh setup
./setup_tracker.sh
```

## Substep 0.2 — Create your source tree

```sh
mkdir -p learn/src/{metrics,logging,gpu,config,pipeline,app}
mkdir -p learn/tests learn/steps
touch learn/src/metrics/__init__.py
touch learn/src/logging/__init__.py
touch learn/src/gpu/__init__.py
touch learn/src/config/__init__.py
touch learn/src/pipeline/__init__.py
```

## Substep 0.3 — Python path

Inside the container, add `learn/src` to imports:

```sh
export PYTHONPATH=/workspace/learn/src:$PYTHONPATH
```

Then `from metrics.stage_metrics import StageMetrics` works in steps and tests.

## Substep 0.4 — Sample URI for short runs

Default clip (inside container):

```text
file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```

## Checkpoint

```sh
docker exec ds8-dev test -f /workspace/models/yolo26m.onnx && echo OK
```

Proceed to [01-stage_metrics.md](01-stage_metrics.md).
