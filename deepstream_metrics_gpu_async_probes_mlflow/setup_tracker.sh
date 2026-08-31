#!/bin/sh
# Compile YOLO26 parser, download Market-1501 ReID + YOLO26m, export ONNX.
#   ./setup_tracker.sh
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$DIR/dev.sh" up >/dev/null
exec docker exec -i ds8-dev python3 /workspace/setup_tracker.py
