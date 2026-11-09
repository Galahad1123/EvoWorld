"""CPU-only verification of real original SAGE outputs; never relabel or edit HDF5."""
import contextlib
import hashlib
import io
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


def review(run, root):
    run = Path(run)
    result = {'run': str(run), 'passed': False, 'checks': {}}
    checks = result['checks']
    log = (run / 'run.log').read_text()
    marker = re.search(r'SAGE_ORIGINAL_LOOP_COMPLETE attempts=(\d+) demos=(\d+)', log)
    checks['loop_completed'] = marker is not None
    checks['no_python_traceback'] = 'Traceback (most recent call last)' not in log
    result['attempts'] = int(marker[1]) if marker else None
    result['collected_demos'] = int(marker[2]) if marker else 0
    checks['original_success_collected'] = result['collected_demos'] > 0
    checks['m2t2_ran'] = bool(re.search(r'object_\d+ has [1-9]\d* grasps', log))
    checks['curobo_planned_approach_and_place'] = (
        'planning place traj' in log and log.count('Trajectory Generated: success True') >= 2)
    files = list(run.rglob('*.hdf5'))
    checks['one_hdf5'] = len(files) == 1
    if len(files) == 1:
        path = files[0]
        result['hdf5'] = str(path.relative_to(run))
        result['hdf5_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with h5py.File(path) as f:
            episodes = [k for k in f['data'] if k.startswith('demo_')]
            result['episodes'] = episodes
            checks['episode_count_matches'] = len(episodes) == result['collected_demos'] > 0
            checks['sample_total_matches'] = int(f['data'].attrs['total']) == sum(
                int(f['data'][k].attrs['num_samples']) for k in episodes)
            checks['valid_datasets'] = bool(episodes)
            result['episode_details'] = []
            for name in episodes:
                e = f['data'][name]
                n = int(e.attrs['num_samples'])
                action = e['actions'][:]
                detail = {'name': name, 'samples': n, 'obs': {}, 'alignment_max_abs': {}}
                valid = n > 1 and action.shape == (n, 8) and np.isfinite(action).all()
                valid &= bool(np.allclose(np.linalg.norm(action[:, 3:7], axis=1), 1, atol=1e-4))
                valid &= bool(set(np.unique(action[:, 7])).issubset({-1., 1.}))
                valid &= e['rewards'].shape == e['dones'].shape == (n,)
                valid &= bool(np.isfinite(e['rewards'][:]).all())
                valid &= set(e['obs']) == set(e['next_obs'])
                valid &= {'joint_pos', 'joint_vel', 'ee_pose', 'points', 'rgb_left', 'rgb_right',
                          'rgb_wrist', 'depth_left', 'depth_right', 'depth_wrist'} <= set(e['obs'])
                for key in e['obs']:
                    a, b = e['obs'][key][:], e['next_obs'][key][:]
                    valid &= a.shape == b.shape and a.shape[0] == n
                    valid &= bool(np.isfinite(a).all() and np.isfinite(b).all())
                    detail['obs'][key] = {'shape': list(a.shape), 'dtype': str(a.dtype)}
                    # Upstream points() independently samples a point cloud on every compute().
                    # Point ordering is stochastic; exact adjacent-array equality is inapplicable.
                    if key != 'points':
                        delta = float(np.max(np.abs(a[1:].astype(float) - b[:-1].astype(float))))
                        detail['alignment_max_abs'][key] = delta
                        valid &= delta < 1e-5
                    else:
                        valid &= a.shape[1:] == (10000, 3)
                valid &= float(np.max(np.ptp(e['obs/joint_pos'][:], axis=0))) > .05
                checks['valid_datasets'] &= bool(valid)
                detail['dones_true'] = int(np.count_nonzero(e['dones'][:]))
                detail['reward_values'] = np.unique(e['rewards'][:]).tolist()
                result['episode_details'].append(detail)
                canvas = Image.new('RGB', (384, 3 * 148))
                draw = ImageDraw.Draw(canvas)
                for row, idx in enumerate([0, n // 2, n - 1]):
                    for col, key in enumerate(['rgb_left', 'rgb_right', 'rgb_wrist']):
                        canvas.paste(Image.fromarray(e['obs'][key][idx]), (128 * col, 148 * row))
                    draw.text((4, 148 * row + 130), f'{name} step {idx}: left / right / wrist', fill='white')
                canvas.save(run / (name + '-observations.png'))
        # Execute the unchanged upstream consumer, replacing only its hardcoded filename in memory.
        reader = root / 'upstream/sage/server/isaaclab/read_pick_n_place_data.py'
        code = re.sub(r'^file_path = .*$', 'file_path = ' + repr(str(path)), reader.read_text(), count=1, flags=re.M)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(code, str(reader), 'exec'), {})
        (run / 'upstream-reader.log').write_text(output.getvalue())
        checks['original_reader'] = f'Total number of episodes: {result["collected_demos"]}' in output.getvalue()
        result['upstream_reader_sha256'] = hashlib.sha256(reader.read_bytes()).hexdigest()
    diagnostic = run / 'diagnostics.jsonl'
    checks['diagnostic_trace_present'] = diagnostic.is_file()
    if diagnostic.is_file():
        rows = [json.loads(line) for line in diagnostic.read_text().splitlines()]
        # This verifier intentionally targets the original bounded single-env task (400 steps).
        attempts = sorted({r['attempt'] for r in rows})
        checks['complete_bounded_trace'] = (
            marker is not None and attempts == list(range(result['attempts']))
            and all([x['iteration'] for x in rows if x['attempt'] == attempt] == list(range(0, 400, 5))
                    for attempt in attempts))
        successful = [attempt for attempt in attempts
                      if any(any(x.get('placed', [])) for x in rows if x['attempt'] == attempt)]
        checks['trace_matches_collected_episode'] = False
        if len(files) == 1:
            with h5py.File(files[0]) as f:
                episodes = sorted((k for k in f['data'] if k.startswith('demo_')),
                                  key=lambda x: int(x.split('_')[1]))
                matched = len(successful) == len(episodes) == result['collected_demos'] > 0
                for attempt, name in zip(successful, episodes):
                    e = f['data'][name]
                    n = len(e['actions'])
                    # Original collector begins at control iteration T_init + 1 = 21.
                    samples = [x for x in rows if x['attempt'] == attempt and 21 <= x['iteration'] < 21 + n]
                    matched &= len(samples) >= n // 5
                    for x in samples:
                        i = x['iteration'] - 21
                        matched &= bool(np.allclose(x['action'][0], e['actions'][i], atol=1e-6, rtol=0))
                        # joint_qpos contains absolute arm q; joint_pos is relative to defaults.
                        matched &= bool(np.allclose(x['joint_pos'][0][:7], e['next_obs/joint_qpos'][i, :7],
                                                    atol=1e-6, rtol=0))
                checks['trace_matches_collected_episode'] = bool(matched)
        summaries = []
        for attempt in sorted({r['attempt'] for r in rows}):
            r = [x for x in rows if x['attempt'] == attempt and 'initial_z' in x]
            if not r:
                continue
            pos = np.array([x['object_state'][0][:3] for x in r])
            tail = [x for x in r if x['iteration'] >= r[-1]['iteration'] - 50]
            summaries.append({
                'attempt': attempt, 'sampled_max_lift_m': float(pos[:, 2].max() - r[0]['initial_z']),
                'required_lift_m': r[0]['required_lift'],
                'planar_displacement_m': float(np.linalg.norm(pos[-1, :2] - pos[0, :2])),
                'final_com_position_m': pos[-1].tolist(),
                'sampled_tail_speed_max_m_s': float(max(np.linalg.norm(x['object_state'][0][7:10]) for x in tail)),
                'tail_samples': len(tail), 'sample_period_control_steps': 5,
                'placed': any(any(x['placed']) for x in r),
                'final_gripper_joint_positions_m': r[-1]['joint_pos'][0][-2:],
            })
        result['motion_evidence'] = summaries
        checks['actual_lift_and_release'] = any(
            s['placed'] and s['sampled_max_lift_m'] > s['required_lift_m']
            and min(s['final_gripper_joint_positions_m']) > .035 for s in summaries)
    result['limitations'] = [
        'Fixed-base Franka, one small fixture; not all SAGE robots/tasks or statistical success rate.',
        'Original success criterion is used. Diagnostic tail speeds are sampled, not a new full-rate stability certification.',
        'Depth observations are upstream uint8 preprocessing; they are not metric depth.',
        'Point clouds are independently sampled; array-wise temporal equality is not expected.',
        'Original task has no reward/termination terms: dones remain false and rewards zero; collector success is separate.',
        'Readback and schema validation do not claim a completed training run.',
    ]
    result['passed'] = all(checks.values())
    return result


if __name__ == '__main__':
    run = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[2]
    try:
        report = review(run, root)
    except Exception as error:
        import traceback
        report = {'passed': False, 'error': str(error), 'traceback': traceback.format_exc()}
    (run / 'reviewed-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
