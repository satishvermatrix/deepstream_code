#!/bin/sh
# Sweep N copies of the sample file through the tracker. First N>1 run builds
# a TensorRT engine for that batch (slow). Later sweeps reuse engines.
#   ./sweep_tracker_n.sh
#   ./sweep_tracker_n.sh --streams 1,2,4
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$DIR/dev.sh" up >/dev/null
exec docker exec -i ds8-dev python3 /workspace/sweep_tracker_n.py "$@"
