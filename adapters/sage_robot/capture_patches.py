"""Capture all copied tracked-source edits and verify pristine upstream sources."""
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
out = root / 'patches/sage-robot'
out.mkdir(parents=True, exist_ok=True)
manifest = {}
for name, components in [('sage', ['IsaacLab', 'M2T2', 'server']), ('curobo', ['.'])]:
    upstream = root / 'upstream' / name
    status = subprocess.check_output(['git', '-C', str(upstream), 'diff', '--name-only', 'HEAD'], text=True)
    assert not status.strip(), f'Upstream tracked changes: {name}: {status}'
    files = subprocess.check_output(['git', '-C', str(upstream), 'ls-files', '-z'], text=True).split('\0')
    patch, changed = [], []
    for filename in files:
        if not filename or (name == 'sage' and filename.split('/')[0] not in components):
            continue
        original = upstream / filename
        relative = Path(filename) if name == 'sage' else Path('curobo') / filename
        copied = root / 'adapters/sage_robot/src' / relative
        if not original.is_file() or not copied.is_file():
            continue
        old, new = original.read_bytes(), copied.read_bytes()
        if old == new:
            continue
        a, b = old.decode(), new.decode()
        patch.extend(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                        fromfile='a/' + str(relative), tofile='b/' + str(relative)))
        changed.append({'path': str(relative), 'upstream_sha256': hashlib.sha256(old).hexdigest(),
                        'adapted_sha256': hashlib.sha256(new).hexdigest()})
    patch_path = out / (name + '-compat.patch')
    patch_path.write_text(''.join(patch))
    manifest[name] = {
        'commit': subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
        'upstream_tracked_clean': True, 'changed_files': changed,
        'patch_sha256': hashlib.sha256(patch_path.read_bytes()).hexdigest(),
    }
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print({name: len(info['changed_files']) for name, info in manifest.items()})
