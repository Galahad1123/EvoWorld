#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
BUNDLE="$(realpath "$1")"
OUTPUT="$(realpath -m "$2")"
test ! -e "$OUTPUT"
source adapters/sage_robot/env.sh
export EVOWORLD_SAGE_FRANKA_ONLY=1
# Snapshot exact adapter/control inputs before startup; metadata may contain local paths, no credentials.
envs/g1-realtosim/bin/python - "$OUTPUT" <<'PY_SNAPSHOT'
import json,hashlib,sys,shutil,subprocess
from pathlib import Path
output=Path(sys.argv[1]);dest=Path(str(output)+'.sources');dest.mkdir()
paths=list(Path('adapters/sage_realtosim').glob('*.py'))+list(Path('adapters/sage_realtosim').glob('*.sh'))+[Path('adapters/single_arm/contract.py')]
hashes={}
for p in paths:
    target=dest/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(dest/'manifest.json').write_text(json.dumps({'timing':'pre-run','files':hashes,'upstream':{n:subprocess.check_output(['git','-C','upstream/'+n,'rev-parse','HEAD'],text=True).strip() for n in ['sage','g1-realtosim']}},indent=2))
PY_SNAPSHOT
# Logs sit next to output, since runtime refuses an existing output directory.
set +e
timeout --signal=TERM --kill-after=15 180 envs/sage-robot/bin/python -u adapters/sage_realtosim/runtime.py --bundle "$BUNDLE" --output "$OUTPUT" --headless --enable_cameras --experience "$PROJECT_ROOT/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit" > "$OUTPUT.log" 2>&1
SIM_EXIT=$?
set -e
printf '%s\n' "$SIM_EXIT" > "$OUTPUT.exit-code.txt"
test "$SIM_EXIT" -eq 0
# Kit may swallow a Python error; require fresh explicit runtime completion too.
envs/g1-realtosim/bin/python - "$OUTPUT" <<'PY'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/'completed.json').read_text());assert r['passed_execution'] and r['exit_code']==0 and r['rows']==160,r
print(sys.argv[1],r)
PY
