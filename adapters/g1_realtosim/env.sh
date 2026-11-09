#!/usr/bin/env bash
# Process-local tools only; no system/environment upgrades or credential loading.
EVOWORLD_G1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$EVOWORLD_G1_ROOT/configs/mirrors.env"
export PATH="$EVOWORLD_G1_ROOT/envs/g1-realtosim/bin:$EVOWORLD_G1_ROOT/tools/ffmpeg-jammy/usr/bin:$PATH"
