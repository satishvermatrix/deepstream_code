#!/bin/sh
# Run pipeline_tracker_multisrc.py inside ds8-dev.
#   ./run_tracker_multisrc.sh -n 4
#   ./run_tracker_multisrc.sh -n 4 --sink file
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$DIR/dev.sh" up >/dev/null
exec docker exec -i ds8-dev python3 /workspace/pipeline_tracker_multisrc.py "$@"
