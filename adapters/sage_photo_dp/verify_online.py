"""CPU audit of saved online traces and decoded videos; no simulator/inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
args = p.parse_args()
out = args.output.resolve()
report = json.loads((out / 'report.json').read_text())
plan = json.loads((out / 'trial-plan.json').read_text())
assert report['complete'] and (out / 'exit-status.txt').read_text().strip() == '0'
expected = 10 if report['mode'] == 'policy' else 2
assert report['total'] == len(report['episodes']) == expected
assert [r['seed'] for r in report['episodes']] == plan['seeds']
h = hashlib.sha256()
with Path(report['checkpoint']).open('rb') as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(chunk)
assert h.hexdigest() == report['checkpoint_sha256'] == plan['sha256']
rows = []
initial_poses = []
for result in report['episodes']:
    trial = result['trial']
    with np.load(out / (trial + '.npz')) as a:
        states, roots, actions, raw = [a[k] for k in ['states', 'roots', 'actions', 'raw_actions']]
        assert len(states) == len(roots) == len(actions) == result['steps']
        assert all(np.isfinite(a[k]).all() for k in a.files)
        assert not result['error'], 'This audit requires complete, valid-action traces'
        assert len(states) == len(raw) + 50
        if report['mode'] == 'policy':
            assert len(raw) == 750
        projected = raw.astype('float32').copy()
        projected[:, 3:7] /= np.linalg.norm(projected[:, 3:7], axis=1)[:, None]
        projected[:, 7] = np.where(projected[:, 7] >= 0, 1, -1)
        np.testing.assert_allclose(actions[:len(raw)], projected, atol=1e-7, rtol=1e-6)
        np.testing.assert_allclose(actions[-50:], np.tile(actions[-51], (50, 1)), atol=0, rtol=0)
        rotation = Rotation.from_quat(roots[-1, [4, 5, 6, 3]])
        support = (rotation.apply(a['corners']) + roots[-1, :3])[:, 2].min() - a['table_z']
        metrics = {
            'lift_m': float(states[:, 2].max() - a['z0']),
            'xy_error_m': float(np.linalg.norm(roots[-1, :2] - a['goal_xy'])),
            'support_error_m': float(support),
            'tilt_deg': float(np.degrees(np.arccos(np.clip(rotation.apply([0, 0, 1])[2], -1, 1)))),
            'tail_speed_m_s': float(np.linalg.norm(states[-25:, 7:10], axis=1).max()),
            'tail_angular_speed_rad_s': float(np.linalg.norm(states[-25:, 10:13], axis=1).max()),
            'finger_opening_m': float(a['joints'][-1, -2:].mean()),
        }
        for key, value in metrics.items():
            np.testing.assert_allclose(value, result['metrics'][key], atol=1e-8, rtol=1e-6)
        checks = dict(zip(['lift', 'placement', 'support', 'upright', 'stable', 'angular_stable', 'released'], [
            metrics['lift_m'] > .03, metrics['xy_error_m'] < .05, abs(support) < .015,
            metrics['tilt_deg'] < 15, metrics['tail_speed_m_s'] < .05,
            metrics['tail_angular_speed_rad_s'] < .2, metrics['finger_opening_m'] > .025]))
        assert checks == result['checks'] and all(checks.values()) == result['success']
    with imageio.get_reader(str(out / (trial + '.mp4'))) as reader:
        metadata = reader.get_meta_data()
        assert metadata['fps'] == 25
        frames = 0
        for frame in reader:
            assert frame.shape == (128, 256, 3)
            frames += 1
    assert frames == 1 + result['steps'] // 2
    with np.load(out / (trial + '-initial.npz')) as initial:
        initial_poses.append(initial['milk_root'][:7].tolist())
    rows.append({'trial': trial, 'success': result['success'], 'decoded_video_frames': frames, 'fps': 25})
if report['mode'] == 'policy':
    assert len(np.unique(np.round(initial_poses, 6), axis=0)) == 10
assert report['successes'] == sum(r['success'] for r in rows)
verification = {'passed': True, 'mode': report['mode'], 'total': expected,
                'successes': report['successes'], 'checks': 'checkpoint SHA, complete traces, action projection/hold, independently recomputed physics, full video decode, distinct initial poses',
                'episodes': rows, 'initial_milk_poses': initial_poses,
                'failed_checks': dict(Counter(k for r in report['episodes'] for k in r['failed_checks']))}
(out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
(out / 'verify_online-executed.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(verification))
