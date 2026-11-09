#!/usr/bin/env bash
set -euo pipefail
EVOWORLD_ROBOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$EVOWORLD_ROBOT_ROOT/adapters/sage_robot/env.sh"
export EVOWORLD_SAGE_FRANKA_ONLY=1
export EVOWORLD_TARGET_OBJECT="${EVOWORLD_TARGET_OBJECT:-room_evoworld_30dd907ee170_mug}"
export EVOWORLD_LAYOUT_ID="${EVOWORLD_LAYOUT_ID:-layout_evoworld_30dd907ee170}"
export EVOWORLD_PLACE_OBJECT=room_evoworld_30dd907ee170_table
export EVOWORLD_TABLE_OBJECT=room_evoworld_30dd907ee170_table
export EVOWORLD_MAX_TRIALS="${EVOWORLD_MAX_TRIALS:-2}"
EVOWORLD_ORIGINAL_RUN="$(mktemp -d "$EVOWORLD_ROBOT_ROOT/experiments/sage-original-task-XXXXXXXX")"
echo "$EVOWORLD_ORIGINAL_RUN"
"$EVOWORLD_ROBOT_ROOT/envs/sage/bin/python" "$EVOWORLD_ROBOT_ROOT/adapters/sage_robot/capture_run.py" "$EVOWORLD_ORIGINAL_RUN" --layout "$EVOWORLD_LAYOUT_ID" --timing pre-run
cd "$EVOWORLD_ORIGINAL_RUN"
# Original collector writes under cwd. Do not load API credentials for the robot-only task.
set +e
timeout --signal=TERM --kill-after=15 300 "$EVOWORLD_ROBOT_ROOT/envs/sage-robot/bin/python" -u \
 "$EVOWORLD_ROBOT_ROOT/adapters/sage_robot/src/server/isaaclab/data_generation_franka_pnp_from_layout_parsing.py" \
 --headless --enable_cameras --num_envs 1 --num_demos 1 --filename evoworld_original \
 --layout_id "$EVOWORLD_LAYOUT_ID" --room_id room_evoworld_30dd907ee170 \
 --experience "$EVOWORLD_ROBOT_ROOT/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" \
 > "$EVOWORLD_ORIGINAL_RUN/run.log" 2>&1

EVOWORLD_SIM_EXIT=$?
set -e
printf '%s\n' "$EVOWORLD_SIM_EXIT" > "$EVOWORLD_ORIGINAL_RUN/simulator-exit-code.txt"
"$EVOWORLD_ROBOT_ROOT/envs/sage/bin/python" "$EVOWORLD_ROBOT_ROOT/adapters/sage_robot/review_run.py" "$EVOWORLD_ORIGINAL_RUN"
test "$EVOWORLD_SIM_EXIT" -eq 0
