#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/adapters/sage_robot/env.sh"
export EVOWORLD_SAGE_FRANKA_ONLY=1
export EVOWORLD_TARGET_OBJECT=room_evoworld_30dd907ee170_cube
export EVOWORLD_LAYOUT_ID=layout_evoworld_cube_control
export EVOWORLD_PLACE_OBJECT=room_evoworld_30dd907ee170_table
export EVOWORLD_TABLE_OBJECT=room_evoworld_30dd907ee170_table
export EVOWORLD_MAX_TRIALS="${EVOWORLD_MAX_TRIALS:-16}"
RUN="$(mktemp -d "$ROOT/experiments/phase8-dp-XXXXXXXX")"
printf '%s\n' "$RUN" | tee "$ROOT/logs/phase8-current.txt"
"$ROOT/envs/sage/bin/python" "$ROOT/adapters/sage_robot/capture_run.py" "$RUN" --layout "$EVOWORLD_LAYOUT_ID" --timing pre-run
cp -r "$ROOT/adapters/sage_dp" "$RUN/adapter-snapshot"
cd "$RUN"
set +e
timeout --signal=TERM --kill-after=15 "${EVOWORLD_COLLECTION_TIMEOUT:-900}" "$ROOT/envs/sage-robot/bin/python" -u "$ROOT/adapters/sage_dp/collect.py" \
 --headless --enable_cameras --num_envs 1 --num_demos "${EVOWORLD_NUM_DEMOS:-8}" --filename experts \
 --layout_id "$EVOWORLD_LAYOUT_ID" --room_id room_evoworld_30dd907ee170 \
 --experience "$ROOT/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" > collection.log 2>&1
EXIT=$?
set -e
printf '%s\n' "$EXIT" > collection-exit-code.txt
test "$EXIT" -eq 0
grep 'SAGE_ORIGINAL_LOOP_COMPLETE' collection.log
