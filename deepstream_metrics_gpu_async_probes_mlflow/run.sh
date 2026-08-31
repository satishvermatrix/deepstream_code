#!/bin/sh
# Run pipeline_detector.py inside ds8-dev.
#   ./run.sh
#   ./run.sh --log-mode probe --run-name log_probe
#   ./run.sh --log-mode async --run-name log_async
#   ./run.sh --profile
set -eu
DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$DIR/dev.sh" up >/dev/null
docker exec -i ds8-dev python3 /workspace/pipeline_detector.py "$@"
status=$?

profile=0
for arg in "$@"; do
  [ "$arg" = "--profile" ] && profile=1
done
if [ "$profile" = 1 ] && [ "$status" -eq 0 ]; then
  latest=$(ls -1t "$DIR"/output/nsys/*.nsys-rep 2>/dev/null | head -1 || true)
  if [ -n "${latest:-}" ]; then
    base=${latest%.nsys-rep}
    echo "nsys stats -> ${base}_nvtx.csv"
    docker exec -i ds8-dev nsys stats --report nvtx_sum --format csv \
      "/workspace/output/nsys/$(basename "$latest")" \
      > "${base}_nvtx.csv" || true
    docker exec -i ds8-dev python3 /workspace/compare_log_modes.py --attach-nsys \
      "/workspace/output/nsys/$(basename "${base}_nvtx.csv")" || true
  fi
fi
exit "$status"
