# Source this from any directory. Independent RLinf/OpenPI environment.
EVOWORLD_RLINF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$EVOWORLD_RLINF_ROOT/envs/rlinf-py312/bin/activate"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$EVOWORLD_RLINF_ROOT/upstream/RLinf"
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export OMNI_KIT_ACCEPT_EULA=YES
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
export EMBODIED_PATH="$EVOWORLD_RLINF_ROOT/upstream/RLinf/examples/embodiment"
export REPO_PATH="$EVOWORLD_RLINF_ROOT/upstream/RLinf"

export SLURM_JOB_ID="${SLURM_JOB_ID:-evoworld-rlinf}"

export EVOWORLD_ISAAC_EXPERIENCE="$EVOWORLD_RLINF_ROOT/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit"
export EVOWORLD_ISAAC_KIT_ARGS="--/isaaclab/cameras_enabled=true"

export PYTHONUNBUFFERED=1
