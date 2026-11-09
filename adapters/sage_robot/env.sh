#!/usr/bin/env bash
# Source from EvoWorld root. All changes are scoped to this shell/process.
EVOWORLD_ROBOT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$EVOWORLD_ROBOT_ROOT/configs/mirrors.env"
export CUDA_HOME="$EVOWORLD_ROBOT_ROOT/envs/sage-robot/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$EVOWORLD_ROBOT_ROOT/envs/sage-robot/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
export CPATH="$("$EVOWORLD_ROBOT_ROOT/envs/sage-robot/bin/python" - "$EVOWORLD_ROBOT_ROOT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1]);paths=[]
for name in ['sage-robot','isaacsim6']:
 p=root/'envs'/name/'lib/python3.12/site-packages/nvidia'
 paths.extend(str(x) for x in p.glob('*/include') if x.is_dir())
 cccl=p/'cu13/include/cccl'
 if cccl.is_dir():paths.append(str(cccl))
print(':'.join(paths))
PY
)${CPATH:+:$CPATH}"
export PYTHONPATH="$EVOWORLD_ROBOT_ROOT/adapters/sage_robot/src/IsaacLab/source/extensions/omni.isaac.lab:$EVOWORLD_ROBOT_ROOT/adapters/sage_robot/src/IsaacLab/source/extensions/omni.isaac.lab_assets:$EVOWORLD_ROBOT_ROOT/adapters/sage_robot/src/IsaacLab/source/extensions/omni.isaac.lab_tasks${PYTHONPATH:+:$PYTHONPATH}"

export EVOWORLD_SAGE_ASSET_ROOT="$EVOWORLD_ROBOT_ROOT/assets/isaac42"
