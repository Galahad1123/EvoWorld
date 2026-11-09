"""Bounded CPU installation verification; no rendering, control, training or API calls."""
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

root = Path(__file__).resolve().parents[2]
repo = root / 'upstream/g1-realtosim'
out = Path(tempfile.mkdtemp(prefix='phase4-g1-install-', dir=root / 'experiments'))
print(out, flush=True)
env = os.environ.copy()
env['PATH'] = str(root / 'tools/ffmpeg-jammy/usr/bin') + os.pathsep + env['PATH']
started = time.monotonic()
records = []
report = {'passed': False, 'scope': 'standalone CPU installation', 'commands': records}


def run(name, args):
    t = time.monotonic()
    p = subprocess.run([str(x) for x in args], env=env, cwd=root, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    (out / (name + '.stdout')).write_text(p.stdout)
    (out / (name + '.stderr')).write_text(p.stderr)
    records.append({'name': name, 'argv': [str(x) for x in args], 'returncode': p.returncode,
                    'wall_seconds': time.monotonic() - t})
    if p.returncode:
        raise RuntimeError(f'{name}: exit {p.returncode}; inspect preserved logs')
    return p.stdout


try:
    sha = run('git-sha', ['git', '-C', repo, 'rev-parse', 'HEAD']).strip()
    assert sha == 'eb80bd505a749b367d129a9283c672b8cc34317c'
    run('upstream-clean', ['git', '-C', repo, 'diff', '--exit-code', 'HEAD'])
    cli = [sys.executable, '-m', 'g1_realtosim']
    run('cli-help', cli + ['--help'])
    doctor = json.loads(run('doctor', cli + ['--runtime', repo / 'configs/local.evoworld.runtime.json', 'doctor']))
    assert all(doctor['modules'][k] for k in ['numpy', 'h5py', 'pxr'])
    assert all(doctor['tools'][k] for k in ['ffmpeg', 'ffprobe', 'git'])
    isaac = Path(doctor['configured_interpreters']['isaac_python'])
    assert isaac.is_file()
    run('configured-isaac-interpreter', [isaac, '-c',
        'import sys,importlib.util,importlib.metadata as m,json; assert importlib.util.find_spec("isaacsim"); '
        'print(json.dumps({"executable":sys.executable,"isaacsim":m.version("isaacsim")}))'])
    run('upstream-tests', [sys.executable, '-m', 'unittest', 'discover', '-s', repo / 'tests', '-v'])
    test_log = (out / 'upstream-tests.stderr').read_text()
    test_match = re.search(r'^Ran (\d+) tests? in ', test_log, re.M)
    assert test_match and int(test_match[1]) > 0, 'No upstream tests discovered; refusing empty success'
    test_count = int(test_match[1])
    run('dependency-check', [Path.home() / '.local/bin/uv', 'pip', 'check', '--python', sys.executable])
    run('ffmpeg-version', ['ffmpeg', '-version'])
    run('ffprobe-version', ['ffprobe', '-version'])
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Cube.Define(stage, '/InstallationProbe').CreateSizeAttr(.05)
    stage.GetRootLayer().Export(str(out / 'usd-binding-probe.usda'))
    reopened = Usd.Stage.Open(str(out / 'usd-binding-probe.usda'))
    assert reopened.GetPrimAtPath('/InstallationProbe').IsValid()
    assert UsdGeom.GetStageMetersPerUnit(reopened) == 1.
    task = root / 'experiments/sage-original-task-LlLIdyMg'
    image = task / 'task-preview.png'
    video = next(task.rglob('000_cameras.mp4'))
    for kind, source in [('image', image), ('video', video)]:
        dest = out / ('ingest-' + kind)
        run('ingest-' + kind, cli + ['ingest', source, dest, '--interval', '.5', '--max-frames', '2'])
        manifest = json.loads((dest / 'inputs.json').read_text())
        item = manifest['media'][0]
        assert len(manifest['media']) == 1 and item['kind'] == kind
        assert item['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert len(item['references']) == (1 if kind == 'image' else 2)
        assert all((dest / f).is_file() for f in item['references'])
        assert json.loads((dest / 'reconstruction_plan.json').read_text())['status'] == 'awaiting_agent_reconstruction'
    report.update(passed=True, commit=sha, doctor=doctor, usd_version=Usd.GetVersion(),
                  input_kind='Existing SAGE simulation preview/video, NOT real-world photos or new reconstruction',
                  tests=test_count, external_dependencies={
                      'isaac6': 'existing interpreter present; g1 renderer GPU compatibility not tested',
                      'isaaclab': 'g1-compatible PT128 environment not installed/validated',
                      'blender': 'not configured; needed for later Blender recipes',
                      'lerobot_av': 'optional exporter not installed',
                      'hunyuan_sonic_grail': 'not downloaded; not required by standalone CPU checks',
                      'g1_trajectory_and_matching_usd': 'not available; no G1 render/control smoke run'},
                  packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()})
except BaseException:
    report['error'] = traceback.format_exc()
    traceback.print_exc()
finally:
    report['wall_seconds'] = time.monotonic() - started
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print('PHASE4_CPU_INSTALL_PASS' if report['passed'] else 'PHASE4_CPU_INSTALL_FAIL', flush=True)
raise SystemExit(0 if report['passed'] else 1)
