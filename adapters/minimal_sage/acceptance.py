"""Require persisted evidence; Isaac shutdown alone is not evidence of success."""
import json

def checked_report(scene_dir,returncode,log_name,report_name,marker):
    if returncode!=0 or list(scene_dir.glob('*-error.json')):
        raise RuntimeError('Simulator subprocess failed; inspect the saved log/error report')
    if not (scene_dir/log_name).is_file() or marker not in (scene_dir/log_name).read_text():
        raise RuntimeError('Simulator did not emit its required completion marker')
    if not (scene_dir/report_name).is_file():
        raise RuntimeError('Simulator did not write the required result artifact')
    report=json.loads((scene_dir/report_name).read_text())
    if not isinstance(report,dict) or not isinstance(report.get('passed'),bool):
        raise RuntimeError('Malformed simulator result')
    return report

# One MCP tool can run physics and rendering sequentially.
SUBPROCESS_TIMEOUT_SECONDS = 300
MCP_TIMEOUT_SECONDS = 2 * SUBPROCESS_TIMEOUT_SECONDS + 60


def assess_trace_motion(traces, expected_samples=120, tail_samples=10, reference_centers=None):
    import numpy as np
    if expected_samples < 2 or not 2 <= tail_samples <= expected_samples or not traces:
        raise ValueError('Invalid trace/sample window')
    motion_observed=False
    tail_motion_bounded=True
    per_body={}
    for name,trace in traces.items():
        positions=np.asarray(trace['position_traj'],dtype=float)
        quaternions=np.asarray(trace['orientation_traj'],dtype=float)
        if positions.shape!=(expected_samples,3) or quaternions.shape!=(expected_samples,4) or not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
            raise ValueError(f'Expected {expected_samples} finite position/quaternion samples per body')
        norms=np.linalg.norm(quaternions,axis=1)
        if np.any(norms<1e-6): raise ValueError('Invalid zero quaternion')
        quaternions=quaternions/norms[:,None]
        if reference_centers is not None:
            from trimesh.transformations import quaternion_matrix
            center=np.asarray(reference_centers[name])
            positions=np.array([quaternion_matrix(q)[:3,:3]@center+p for q,p in zip(quaternions,positions)])
        steps=np.linalg.norm(np.diff(positions,axis=0),axis=1)
        rotation_steps=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(quaternions[1:]*quaternions[:-1],axis=1)),0,1)))
        extent=float(np.linalg.norm(np.ptp(positions,axis=0)))
        rotation_extent=float(np.degrees(2*np.arccos(np.clip(np.min(np.abs(quaternions@quaternions[0])),0,1))))
        observed=extent>1e-5 or rotation_extent>.001
        tail_translation=float(steps[-(tail_samples-1):].max())
        tail_rotation=float(rotation_steps[-(tail_samples-1):].max())
        bounded=tail_translation<=.001 and tail_rotation<=.1
        motion_observed=motion_observed or observed
        tail_motion_bounded=tail_motion_bounded and bounded
        per_body[name]={'translation_extent_m':extent,'rotation_extent_degrees':rotation_extent,
            'max_tail_translation_step_m':tail_translation,'max_tail_rotation_step_degrees':tail_rotation}
    return {'motion_observed':bool(motion_observed),'tail_motion_bounded':bool(tail_motion_bounded),
        'translation_reference':'mesh bounding-box reference center' if reference_centers is not None else 'prim origin','tail_samples':tail_samples,'tail_translation_limit_m':.001,'tail_rotation_limit_degrees':.1,'per_body':per_body}

def validate_timing(requested_steps,observed_steps,elapsed,callback_dts):
    import math
    target=requested_steps/60.0
    if observed_steps!=requested_steps or len(callback_dts)!=requested_steps:
        raise ValueError('Measured step/callback count differs from the requested budget')
    if not math.isclose(elapsed,target,abs_tol=1e-5) or not math.isclose(sum(callback_dts),target,abs_tol=1e-5):
        raise ValueError('Actual physics elapsed time differs from the requested duration')
    if not all(math.isclose(dt,1/60,abs_tol=1e-7) for dt in callback_dts):
        raise ValueError('Unexpected physics timestep')
    return {'requested_steps':requested_steps,'observed_steps':observed_steps,'callback_count':len(callback_dts),
        'simulation_seconds':elapsed,'callback_seconds':sum(callback_dts),'physics_dt':1/60}

DEFAULT_RUN_CONFIG={'physics_steps':120,'max_turns':10,'max_simulations':3}

def validate_run_config(config):
    if set(config)!=set(DEFAULT_RUN_CONFIG) or any(type(v) is not int for v in config.values()):
        raise ValueError('Invalid run configuration')
    if config['physics_steps'] not in [120,600] or not 4<=config['max_turns']<=10 or not 1<=config['max_simulations']<=3:
        raise ValueError('Run configuration exceeds supported resource limits')
    return dict(config)

def read_run_config(run_dir):
    path=run_dir/'run-config.json'
    return validate_run_config(json.loads(path.read_text()) if path.exists() else DEFAULT_RUN_CONFIG)

def trajectory_limits(trace,reference_center):
    import numpy as np
    from trimesh.transformations import quaternion_matrix
    positions=np.asarray(trace['position_traj'])
    quats=np.asarray(trace['orientation_traj'])
    center=np.asarray(reference_center)
    initial_q=np.asarray(trace['initial_orientation'])
    initial_center=quaternion_matrix(initial_q)[:3,:3]@center+np.asarray(trace['initial_position'])
    centers=np.array([quaternion_matrix(q)[:3,:3]@center+p for q,p in zip(quats,positions)])
    displacement=np.linalg.norm(centers-initial_center,axis=1)
    normalized=quats/np.linalg.norm(quats,axis=1)[:,None]
    initial_q=initial_q/np.linalg.norm(initial_q)
    angles=np.degrees(2*np.arccos(np.clip(np.abs(normalized@initial_q),0,1)))
    max_shift=float(displacement.max());max_rotation=float(angles.max())
    return {'passed':max_shift<=.05 and max_rotation<=8.,'max_reference_center_displacement_m':max_shift,
        'max_rotation_change_degrees':max_rotation,'samples_checked':len(positions)}
