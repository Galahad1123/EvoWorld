"""Fingerprint non-secret run inputs; timing label distinguishes retrospective records."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('--layout', required=True)
p.add_argument('--timing', choices=['pre-run', 'post-run'], required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
fixture = root / 'adapters/sage_robot/src/server/results' / a.layout
manifest = root / 'patches/sage-robot/manifest.json'
files = [manifest, root / 'assets/m2t2/m2t2.pth', root / 'assets/isaac42/sage-franka-manifest.json']
files += list((root / 'adapters/sage_robot').glob('*.py'))
files += list((root / 'adapters/sage_robot').glob('*.sh'))
files += [x for x in fixture.rglob('*') if x.is_file()]
for entry in json.loads(manifest.read_text()).values():
    files += [root / 'adapters/sage_robot/src' / x['path'] for x in entry['changed_files']]
hashes = {str(x.relative_to(root)): hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
record = {
    'captured_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'capture_timing': a.timing, 'layout_id': a.layout,
    'environment': {k: os.environ.get(k) for k in
                    ['EVOWORLD_TARGET_OBJECT', 'EVOWORLD_PLACE_OBJECT', 'EVOWORLD_TABLE_OBJECT', 'EVOWORLD_MAX_TRIALS']},
    'sha256': hashes,
    'note': 'Post-run records cannot prove a pre-run snapshot. Launcher safeguards may have been added afterward.'
            if a.timing == 'post-run' else 'Captured immediately before launching the simulator.',
}
a.run.mkdir(parents=True, exist_ok=True)
(a.run / ('input-provenance-' + a.timing + '.json')).write_text(json.dumps(record, indent=2))
shutil.copy2(manifest, a.run / ('source-manifest-' + a.timing + '.json'))
print('fingerprinted', len(hashes), 'inputs', a.timing)
