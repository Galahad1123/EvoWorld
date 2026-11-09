"""Independent CPU read-back of a bounded asset simulation (not robot demonstrations)."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def validate(run, hdf_path=None):
    out = run / 'physics'
    execution = json.loads((out / 'execution.json').read_text())
    package = json.loads((run / 'package-check.json').read_text())
    assert execution['completed'] and package['passed'] and not (out / 'error.json').exists()
    log = (out / 'process.log').read_text()
    assert 'PHASE5_ASSET_PHYSICS_COMPLETE' in log and 'Traceback (most recent call last)' not in log
    spec = json.loads((run / 'scene.json').read_text())
    block = next(x for x in spec['objects'] if x['id'] == 'block')
    with h5py.File(hdf_path or out / 'asset-physics.hdf5', 'r') as f:
        assert f.attrs['schema'] == 'evoworld.asset_physics.v1'
        assert f.attrs['quaternion'] == 'wxyz' and f.attrs['up_axis'] == 'Z'
        assert f.attrs['units'] == 'm,s,rad'
        assert f.attrs['image_semantics'].startswith('offline frozen-state rerenders after physics stopped')
        names = []
        f.visit(names.append)
        assert not any(n.split('/')[-1] in {'actions', 'rewards', 'dones'} for n in names)
        t, p, v, w = [f[n][:] for n in ['time', 'block/pose_world', 'block/linear_velocity_world', 'block/angular_velocity_world']]
        rgb, idx = f['images/rgb'][:], f['images/state_index'][:]
        cameras, timing = json.loads(f.attrs['cameras']), json.loads(f.attrs['timing'])
    assert timing == execution['timing']
    integration = json.loads((out / 'integration.json').read_text())
    assert timing == integration['timing'] and execution['mass_kg'] == integration['mass_kg']
    with np.load(out / 'integrated-states.npz') as original:
        for key, actual in [('time', t), ('poses', p), ('velocities', v), ('angular', w)]:
            assert np.array_equal(original[key], actual), 'Rendering changed previously recorded physics'
    assert timing['steps'] == timing['callback_count'] == 120
    assert abs(timing['simulation_seconds'] - 2) < 1e-5 and abs(timing['callback_seconds'] - 2) < 1e-5
    assert t.shape == (121,) and p.shape == (121, 7) and v.shape == w.shape == (121, 3)
    assert all(np.isfinite(a).all() for a in [t, p, v, w])
    assert t[0] == 0 and np.allclose(np.diff(t), 1 / 60, atol=1e-7, rtol=0) and abs(t[-1] - 2) < 1e-5
    assert np.allclose(np.linalg.norm(p[:, 3:], axis=1), 1, atol=1e-5)
    assert np.allclose(p[0, :3], execution['initial_position_m'], atol=1e-6)
    assert abs(execution['mass_kg'] - .08) < 1e-6
    rotations = Rotation.from_quat(p[:, [4, 5, 6, 3]])
    half_extent = np.abs(rotations.as_matrix()) @ (np.asarray(block['size_m']) / 2)
    table_bounds = np.asarray(package['bounds_m']['table'])
    support_gap = p[:, 2] - half_extent[:, 2] - table_bounds[1, 2]
    assert abs(support_gap[-1]) < .005, ('Final support gap', support_gap[-1])
    assert support_gap.min() > -.005, ('Table penetration', support_gap.min())
    assert np.all(p[:, :2] - half_extent[:, :2] > table_bounds[0, :2])
    assert np.all(p[:, :2] + half_extent[:, :2] < table_bounds[1, :2])
    drop = p[0, 2] - p[-1, 2]
    assert drop > .02, ('No integrated drop', drop)
    xy_drift = np.max(np.linalg.norm(p[:, :2] - p[0, :2], axis=1))
    assert xy_drift < .02
    tail_speed = np.max(np.linalg.norm(v[-61:], axis=1))
    tail_span = np.linalg.norm(np.ptp(p[-61:, :3], axis=0))
    tail_rotation = np.rad2deg((rotations[-61].inv() * rotations[-61:]).magnitude()).max()
    assert tail_speed < .03 and tail_span < .005 and tail_rotation < 8
    assert rgb.shape == (4, 600, 800, 3) and rgb.dtype == np.uint8
    assert np.array_equal(idx, [0, 120, 120, 120]) and len(cameras) == 4
    assert cameras == json.loads((out / 'camera-metadata.json').read_text())
    reprojection_errors = []
    red_centers = []
    for i, camera in enumerate(cameras):
        assert camera['state_index'] == idx[i] and camera['resolution'] == [800, 600]
        assert np.array_equal(np.asarray(Image.open(out / (camera['name'] + '.png'))), rgb[i])
        matrix = np.asarray(camera['world_T_camera_usd_column_convention'])
        assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
        assert np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
        assert np.allclose(matrix[:3, 3], camera['position_m'], atol=1e-5)
        assert all(camera[k] > 0 for k in ['focal_length_mm', 'horizontal_aperture_mm', 'vertical_aperture_mm'])
        displayed = np.asarray(camera['displayed_block_pose_wxyz'])
        assert np.allclose(displayed[:3], p[idx[i], :3], atol=1e-5)
        assert abs(np.dot(displayed[3:], p[idx[i], 3:])) > 1 - 1e-5
        # This red-cube fixture permits an independent image/pose correspondence test.
        pixels = rgb[i].astype(float)
        mask = (pixels[:, :, 0] > 100) & (pixels[:, :, 0] > 1.5 * pixels[:, :, 1]) & (pixels[:, :, 0] > 1.5 * pixels[:, :, 2])
        ys, xs = np.where(mask)
        assert len(xs) > 100
        center = np.array([xs.mean(), ys.mean()])
        red_centers.append(center)
        camera_point = np.linalg.inv(matrix) @ np.r_[p[idx[i], :3], 1.]
        assert camera_point[2] < 0
        focal_px = camera['focal_length_mm'] / camera['horizontal_aperture_mm'] * 800
        projected = np.array([400 + focal_px * camera_point[0] / -camera_point[2], 300 - focal_px * camera_point[1] / -camera_point[2]])
        error = float(np.linalg.norm(center - projected))
        assert error < 4, ('Rendered block does not match recorded pose', camera['name'], error, center, projected)
        reprojection_errors.append(error)
    assert red_centers[1][1] - red_centers[0][1] > 5, 'Initial drop elevation missing from images'
    assert np.mean(np.abs(rgb[1].astype(float) - rgb[2].astype(float))) > 1
    assert np.mean(np.abs(rgb[1].astype(float) - rgb[3].astype(float))) > .1
    return {'passed': True, 'schema': 'evoworld.asset_physics.v1', 'states': len(t), 'steps': 120,
        'seconds': float(t[-1]), 'drop_m': float(drop), 'final_support_gap_m': float(support_gap[-1]),
        'max_xy_drift_m': float(xy_drift), 'tail_seconds': float(t[-1] - t[-61]),
        'tail_peak_speed_m_s': float(tail_speed), 'tail_position_span_m': float(tail_span),
        'tail_rotation_span_degrees': float(tail_rotation), 'png_matches_hdf5': True,
        'camera_metadata_and_state_indices_valid': True,
        'max_cube_center_reprojection_error_px': max(reprojection_errors),
        'image_semantics': 'offline frozen-state rerenders from integrated trajectory; not live sensor samples',
        'scope': 'Synthetic table/cube asset physics only; no robot controls, policy demonstrations, real calibration, or image-derived dynamics.'}


if __name__ == '__main__':
    import shutil
    import tempfile
    run = Path(sys.argv[1]).resolve()
    report = validate(run)
    rejected = []
    with tempfile.TemporaryDirectory(prefix='phase5-negative-') as tmp:
        for case in ['invalid_time', 'static_pose']:
            bad = Path(tmp) / (case + '.hdf5')
            shutil.copyfile(run / 'physics/asset-physics.hdf5', bad)
            with h5py.File(bad, 'r+') as f:
                if case == 'invalid_time':
                    f['time'][60] = f['time'][59]
                else:
                    f['block/pose_world'][:] = f['block/pose_world'][0]
            try:
                validate(run, bad)
            except AssertionError:
                rejected.append(case)
            else:
                raise AssertionError(f'Accepted corrupt fixture: {case}')
    report['negative_cases_rejected'] = rejected
    (run / 'physics-readback.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
