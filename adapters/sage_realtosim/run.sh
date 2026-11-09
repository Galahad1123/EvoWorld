#!/usr/bin/env bash
# Cached-photo end-to-end benchmark. No API credentials, new model inference or training.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
RUN_DIR="$(mktemp -d "$PROJECT_ROOT/experiments/phase7-integration-XXXXXXXX")"
printf '%s\n' "$RUN_DIR" | tee logs/phase7-current.txt
CPU="$PROJECT_ROOT/envs/g1-realtosim/bin/python"
"$CPU" adapters/sage_realtosim/build.py --output "$RUN_DIR/bundle" > "$RUN_DIR/build.log" 2>&1
"$CPU" adapters/sage_realtosim/verify_bundle.py "$RUN_DIR/bundle" > "$RUN_DIR/bundle-check.json"
bash adapters/sage_realtosim/launch.sh "$RUN_DIR/bundle" "$RUN_DIR/primary"
"$CPU" adapters/sage_realtosim/readback.py "$RUN_DIR/primary" > "$RUN_DIR/primary/readback.log"
"$CPU" - "$RUN_DIR" <<'PY'
from pathlib import Path
import shutil,sys
r=Path(sys.argv[1]);shutil.copytree(r/'bundle',r/'relocated/bundle')
PY
"$CPU" adapters/sage_realtosim/verify_bundle.py "$RUN_DIR/relocated/bundle" > "$RUN_DIR/relocated/bundle-check.json"
bash adapters/sage_realtosim/launch.sh "$RUN_DIR/relocated/bundle" "$RUN_DIR/reload"
"$CPU" adapters/sage_realtosim/readback.py "$RUN_DIR/reload" > "$RUN_DIR/reload/readback.log"
"$CPU" adapters/sage_realtosim/compare_runs.py "$RUN_DIR/primary" "$RUN_DIR/reload" --output "$RUN_DIR/reload-comparison.json"
EVOWORLD_PHASE7_RUN="$RUN_DIR/reload" EVOWORLD_PHASE7_BUNDLE="$RUN_DIR/relocated/bundle" "$CPU" -m unittest discover -s adapters/sage_realtosim -v > "$RUN_DIR/tests-final.log" 2>&1
tail -5 "$RUN_DIR/tests-final.log"
