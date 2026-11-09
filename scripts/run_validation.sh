#!/usr/bin/env bash
set -euo pipefail
EVOWORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$EVOWORLD_ROOT"
source configs/mirrors.env
export EVOWORLD_RUN_DIR="$(mktemp -d "$EVOWORLD_ROOT/experiments/sage-validation-$(date +%Y%m%d-%H%M%S)-XXXX")"
export SLURM_JOB_ID="$(basename "$EVOWORLD_RUN_DIR")"
sage_server_pid=""
cleanup() {
  if [[ -n "$sage_server_pid" ]] && kill -0 "$sage_server_pid" 2>/dev/null; then
    kill -TERM "$sage_server_pid"
    wait "$sage_server_pid" || true
  fi
}
trap cleanup EXIT
echo "RUN_DIR=$EVOWORLD_RUN_DIR"
timeout 180 bash scripts/run_isaac.sh > "$EVOWORLD_RUN_DIR/isaac-physics.log" 2>&1
grep -q EVOWORLD_ISAAC_PHYSICS_PASS "$EVOWORLD_RUN_DIR/isaac-physics.log"
bash scripts/run_isaac.sh --sage --serve --seconds 180 > "$EVOWORLD_RUN_DIR/mcp-server.log" 2>&1 &
sage_server_pid=$!
export EVOWORLD_EXPECTED_SERVER_PID="$sage_server_pid"
ready=false
for ((i=0;i<120;i++)); do
  if grep -q EVOWORLD_SERVER_READY "$EVOWORLD_RUN_DIR/mcp-server.log"; then ready=true; break; fi
  kill -0 "$sage_server_pid" 2>/dev/null || break
  sleep 1
done
[[ "$ready" == true ]]
timeout 60 envs/sage/bin/python -u scripts/test_sage_socket.py > "$EVOWORLD_RUN_DIR/mcp-client.log" 2>&1
grep -q EVOWORLD_SAGE_SOCKET_PASS "$EVOWORLD_RUN_DIR/mcp-client.log"
cleanup
sage_server_pid=""
timeout 300 envs/isaacsim6/bin/python -u scripts/sage_scene_smoke.py > "$EVOWORLD_RUN_DIR/scene-physics.log" 2>&1
grep -q EVOWORLD_SAGE_SCENE_TEST_EXECUTED "$EVOWORLD_RUN_DIR/scene-physics.log"
timeout 180 envs/isaacsim6/bin/python -u scripts/render_sage_scene.py > "$EVOWORLD_RUN_DIR/scene-render.log" 2>&1
grep -q EVOWORLD_SAGE_RELOAD_RENDER_PASS "$EVOWORLD_RUN_DIR/scene-render.log"
"$HOME/.local/bin/uv" pip check --python envs/sage/bin/python > "$EVOWORLD_RUN_DIR/sage-pip-check.log" 2>&1
"$HOME/.local/bin/uv" pip check --python envs/isaacsim6/bin/python > "$EVOWORLD_RUN_DIR/isaac-pip-check.log" 2>&1
echo "SIMULATION_INFRASTRUCTURE_TESTS_PASSED=$EVOWORLD_RUN_DIR"
echo "This does not validate GPT generation, robot tasks, or physical quality of all source assets."
