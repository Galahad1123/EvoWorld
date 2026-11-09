#!/usr/bin/env bash
set -euo pipefail
ROOT="$1"
RUN="$2"
PILOT="$3"
TASK_STAGE=collection
trap 'code=$?; printf "{\"stage\":\"%s\",\"exit_code\":%s}\n" "$TASK_STAGE" "$code" > "$RUN/pipeline-exit.json"' EXIT
cd "$ROOT"
source adapters/sage_robot/env.sh
export EVOWORLD_SAGE_FRANKA_ONLY=1 EVOWORLD_TARGET_OBJECT=milk EVOWORLD_PLACE_OBJECT=desk EVOWORLD_TABLE_OBJECT=desk EVOWORLD_MAX_TRIALS=34 EVOWORLD_DP_SEED=310
unset EVOWORLD_PREVIEW_ONLY
cp -r "$ROOT/adapters/sage_photo_dp" "$RUN/source"
cd "$RUN"
printf '%s\n' "$TASK_STAGE" > stage.txt
set +e
timeout --signal=TERM --kill-after=15 2700 "$ROOT/envs/sage-robot/bin/python" -u "$ROOT/adapters/sage_photo_dp/collect_photo.py" --headless --enable_cameras --num_envs 1 --num_demos 18 --filename experts --layout_id layout_evoworld_cube_control --room_id room_evoworld_30dd907ee170 --experience "$ROOT/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" > collection.log 2>&1
CODE=$?
set -e
printf '%s\n' "$CODE" > collection-exit-code.txt
test "$CODE" -eq 0
TASK_STAGE=dataset_verification
printf '%s\n' "$TASK_STAGE" > stage.txt
env -u PYTHONPATH "$ROOT/envs/g1-realtosim/bin/python" "$ROOT/adapters/sage_photo_dp/prepare_photo.py" "$RUN" "$PILOT" > prepare.log 2>&1
TASK_STAGE=training
printf '%s\n' "$TASK_STAGE" > stage.txt
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout --signal=TERM --kill-after=15 2700 "$ROOT/envs/sage-dp/bin/python" -u "$ROOT/adapters/sage_photo_dp/train_photo.py" "$RUN" > training.log 2>&1
TASK_STAGE=training_complete
printf '%s\n' "$TASK_STAGE" > stage.txt
