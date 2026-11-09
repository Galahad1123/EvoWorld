#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$EVOWORLD_RLINF_ROOT"
# Keep local Ray traffic out of HTTP/SOCKS proxies; only this shell is affected.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY=localhost,127.0.0.1
out="logs/rlinf-setup/check-$(date +%Y%m%d-%H%M%S)"
mkdir "$out"
timeout --kill-after=15 120 python adapters/rlinf/check_runtime.py > "$out/runtime.log" 2>&1
cp logs/rlinf-setup/runtime-check.json "$out/runtime-check.json"
timeout --kill-after=15 180 python adapters/rlinf/check_sim.py --output "$out/sim" > "$out/sim.log" 2>&1
printf 'Both checks passed. Reports: %s/%s\n' "$EVOWORLD_RLINF_ROOT" "$out"
