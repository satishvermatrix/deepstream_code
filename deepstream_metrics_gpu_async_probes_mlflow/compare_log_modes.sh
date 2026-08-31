#!/bin/sh
# A/B: same video, probe logging vs async logging.
#   ./compare_log_modes.sh           # print table from last MLflow runs
#   ./compare_log_modes.sh --run     # run both modes then print the table
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ "${1:-}" = "--run" ]; then
  "$DIR/run.sh" --log-mode probe --run-name log_probe
  "$DIR/run.sh" --log-mode async --run-name log_async
fi
"$DIR/dev.sh" up >/dev/null
exec docker exec -i ds8-dev python3 /workspace/compare_log_modes.py
