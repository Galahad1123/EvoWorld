#!/usr/bin/env bash
# Authoritative export entry: Kit fast shutdown may mask Python exception exit codes.
set -euo pipefail
EVOWORLD_ROBOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$EVOWORLD_ROBOT_ROOT/adapters/sage_robot/env.sh"
export EVOWORLD_LAYOUT_ID="${EVOWORLD_LAYOUT_ID:-layout_evoworld_30dd907ee170}"
EVOWORLD_EXPORT_LOG="$(mktemp "$EVOWORLD_ROBOT_ROOT/logs/phase3-original-export-XXXXXXXX.log")"
echo "$EVOWORLD_EXPORT_LOG"
timeout --signal=TERM --kill-after=15 60 "$EVOWORLD_ROBOT_ROOT/envs/sage-robot/bin/python" \
  "$EVOWORLD_ROBOT_ROOT/adapters/sage_robot/export_fixture.py" > "$EVOWORLD_EXPORT_LOG" 2>&1
grep -q '^SAGE_ORIGINAL_USDZ_PASS$' "$EVOWORLD_EXPORT_LOG"
"$EVOWORLD_ROBOT_ROOT/envs/sage/bin/python" - "$EVOWORLD_ROBOT_ROOT" "$EVOWORLD_LAYOUT_ID" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])/'adapters/sage_robot/src/server/results'/sys.argv[2]/'export-report.json'
assert json.loads(p.read_text())['passed'] is True
PY
