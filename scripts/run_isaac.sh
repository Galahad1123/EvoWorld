#!/usr/bin/env bash
set -euo pipefail
EVOWORLD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$EVOWORLD_ROOT/configs/mirrors.env"
export SAGE_EXT_ROOT="$EVOWORLD_ROOT/adapters/isaac6"
export SAGE_SERVER_ROOT="$EVOWORLD_ROOT/upstream/sage/server"
cd "$EVOWORLD_ROOT"
exec "${EVOWORLD_ISAAC_ENV:-envs/isaacsim6}/bin/python" -u scripts/isaac_entry.py "$@"
