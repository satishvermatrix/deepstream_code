#!/bin/sh
# Run pipeline_tracker.py inside ds8-dev.
#   ./setup_tracker.sh          # first time: models + parser
#   ./run_tracker.sh
#   ./run_tracker.sh --run-name yolo26_nvdcf
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$DIR/dev.sh" up >/dev/null
exec docker exec -i ds8-dev python3 /workspace/pipeline_tracker.py "$@"
