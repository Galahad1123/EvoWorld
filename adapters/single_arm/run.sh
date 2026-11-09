#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
source adapters/sage_robot/env.sh
export EVOWORLD_SAGE_FRANKA_ONLY=1
RUN_DIR="$(mktemp -d "$PROJECT_ROOT/experiments/phase6-single-arm-XXXXXXXX")"
printf '%s\n' "$RUN_DIR" | tee logs/phase6-current.txt
CONFIG_PATH="$PROJECT_ROOT/experiments/sage-original-task-LlLIdyMg/robomimic_data/Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0-vis-teaser/params/env_init.yaml"
envs/g1-realtosim/bin/python - "$RUN_DIR" "$CONFIG_PATH" <<'PY'
import hashlib,json,sys,subprocess
from pathlib import Path
root=Path.cwd();paths=list((root/'adapters/single_arm').glob('*.py'))+list((root/'adapters/single_arm').glob('*.sh'))+[Path(sys.argv[2])]
paths+=list((root/'adapters/sage_robot/src/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/lift_single_obj_scene').rglob('*.py'))
paths += [root/'adapters/sage_robot/src/IsaacLab/source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets/franka.py',root/'adapters/sage_robot/src/IsaacLab/source/extensions/omni.isaac.lab/omni/isaac/lab/envs/mdp/observations.py']
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
report={'timing':'pre-run','files':files,'upstream':{n:{'head':subprocess.check_output(['git','-C',str(root/'upstream'/n),'rev-parse','HEAD'],text=True).strip(),'tracked_changes':subprocess.check_output(['git','-C',str(root/'upstream'/n),'status','--porcelain','--untracked-files=no'],text=True)} for n in ['sage','g1-realtosim']}}
(Path(sys.argv[1])/'provenance-pre-run.json').write_text(json.dumps(report,indent=2))
PY
envs/g1-realtosim/bin/python -m unittest discover -s adapters/single_arm -p test_contract.py -v > "$RUN_DIR/cpu-before.log" 2>&1
set +e
timeout --signal=TERM --kill-after=15 180 envs/sage-robot/bin/python -u adapters/single_arm/audit.py --output "$RUN_DIR" --config "$CONFIG_PATH" --headless --enable_cameras --experience "$PROJECT_ROOT/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" > "$RUN_DIR/run.log" 2>&1
SIM_EXIT=$?
set -e
printf '%s\n' "$SIM_EXIT" > "$RUN_DIR/simulator-exit-code.txt"
test "$SIM_EXIT" -eq 0
envs/g1-realtosim/bin/python adapters/single_arm/readback.py "$RUN_DIR" > "$RUN_DIR/readback.log"
envs/g1-realtosim/bin/python adapters/single_arm/map_scenes.py "$RUN_DIR" > "$RUN_DIR/mapping.log"
EVOWORLD_PHASE6_ACCEPTED="$RUN_DIR" envs/g1-realtosim/bin/python -m unittest discover -s adapters/single_arm -v > "$RUN_DIR/cpu-tests.log" 2>&1
tail -5 "$RUN_DIR/cpu-tests.log"
cat "$RUN_DIR/mapping.log"
