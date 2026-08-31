#!/usr/bin/env bash
# Cursor-on-host / run-in-image helper for DeepStream 8 on this GB10 machine.
#
#   ./dev.sh up      start (or create) container ds8-dev
#   ./dev.sh setup   install official pyds wheel inside the container
#   ./dev.sh shell   interactive shell in the image
#   ./dev.sh status  show whether ds8-dev is running
#   ./dev.sh down    stop the container (keeps it; next "up" is fast)
#   ./dev.sh rm      stop and remove the container
#
# This folder is bind-mounted at /workspace. Edits in Cursor are visible
# immediately; no image rebuild.

set -euo pipefail

NAME="ds8-dev"
IMAGE="nvcr.io/nvidia/deepstream:8.0-triton-dgx-spark"
HOST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Official NVIDIA pyds for DeepStream 8.0 + Python 3.12 + aarch64.
# Do not "pip install pyds" from PyPI — that is a different package.
PYDS_WHEEL="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.2/pyds-1.2.2-cp312-cp312-linux_aarch64.whl"

usage() {
  sed -n '2,12p' "$0"
}

container_exists() {
  docker inspect "$NAME" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)" == "true" ]]
}

cmd_up() {
  if container_running; then
    echo "$NAME is already running"
    return 0
  fi
  if container_exists; then
    echo "Starting existing $NAME"
    docker start "$NAME" >/dev/null
  else
    echo "Creating $NAME from $IMAGE"
    echo "  host:  $HOST_DIR"
    echo "  mount: /workspace"
    docker run -d \
      --name "$NAME" \
      --gpus all \
      --runtime nvidia \
      --shm-size=1g \
      --entrypoint bash \
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
      -e NVIDIA_VISIBLE_DEVICES=all \
      -v "$HOST_DIR":/workspace \
      -w /workspace \
      "$IMAGE" \
      -c "sleep infinity" >/dev/null
  fi
  echo "Ready. GPU check:"
  docker exec "$NAME" nvidia-smi -L
}

cmd_setup() {
  if ! container_running; then
    cmd_up
  fi
  echo "Installing pyds 1.2.2 (DeepStream 8.0 bindings) inside $NAME"
  if docker exec "$NAME" python3 -c "import pyds" 2>/dev/null; then
    echo "pyds already importable"
  else
    docker exec "$NAME" pip3 install --break-system-packages "$PYDS_WHEEL"
  fi
  docker exec "$NAME" python3 -c "import pyds; print('pyds OK', pyds)"
  echo "Installing mlflow (numpy<2 — pyds does not support numpy 2.x)"
  docker exec "$NAME" python3 -c "import mlflow" 2>/dev/null || \
    docker exec "$NAME" pip3 install --break-system-packages --ignore-installed blinker 'numpy<2' 'mlflow'
  docker exec "$NAME" python3 -c "import mlflow; print('mlflow OK', mlflow.__version__)"
}

cmd_shell() {
  if ! container_running; then
    cmd_up
  fi
  exec docker exec -it "$NAME" bash
}

cmd_status() {
  if container_running; then
    echo "$NAME: running"
    docker exec "$NAME" bash -c 'echo "  DS_VERSION=$DS_VERSION"; python3 -c "import pyds" 2>/dev/null && echo "  pyds: OK" || echo "  pyds: NOT INSTALLED (run ./dev.sh setup)"'
  elif container_exists; then
    echo "$NAME: stopped (./dev.sh up to start)"
  else
    echo "$NAME: does not exist (./dev.sh up to create)"
  fi
}

cmd_down() {
  if container_running; then
    docker stop "$NAME" >/dev/null
    echo "Stopped $NAME"
  else
    echo "$NAME is not running"
  fi
}

cmd_rm() {
  if container_exists; then
    docker rm -f "$NAME" >/dev/null
    echo "Removed $NAME"
  else
    echo "$NAME does not exist"
  fi
}

case "${1:-}" in
  up) cmd_up ;;
  setup) cmd_setup ;;
  shell) cmd_shell ;;
  status) cmd_status ;;
  down) cmd_down ;;
  rm) cmd_rm ;;
  -h|--help|help|"") usage ;;
  *) echo "Unknown command: $1"; usage; exit 1 ;;
esac
