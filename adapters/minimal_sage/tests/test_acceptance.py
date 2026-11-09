import json
import pytest
from acceptance import checked_report

def test_exit_zero_without_outputs_does_not_pass(tmp_path):
    (tmp_path/'job.log').write_text('Simulation App Shutting Down')
    with pytest.raises(RuntimeError): checked_report(tmp_path,0,'job.log','physics.json','EXPECTED_MARKER')

def test_error_artifact_overrides_success_marker(tmp_path):
    (tmp_path/'job.log').write_text('EXPECTED_MARKER')
    (tmp_path/'physics.json').write_text(json.dumps({'passed':True}))
    (tmp_path/'physics-error.json').write_text('{}')
    with pytest.raises(RuntimeError): checked_report(tmp_path,0,'job.log','physics.json','EXPECTED_MARKER')

def test_physics_failure_is_not_promoted_to_pass(tmp_path):
    (tmp_path/'job.log').write_text('EXPECTED_MARKER')
    (tmp_path/'physics.json').write_text(json.dumps({'passed':False,'body_count':4}))
    assert checked_report(tmp_path,0,'job.log','physics.json','EXPECTED_MARKER')['passed'] is False

from acceptance import assess_trace_motion

def traces(kind):
    positions=[[0.,0.,0.] for _ in range(120)]
    if kind=='settling':
        for i in range(20): positions[i][2]=.01*(1-i/20)
    if kind=='drifting':
        for i in range(120): positions[i][0]=i*.002
    return {'body':{'position_traj':positions,'orientation_traj':[[1.,0.,0.,0.] for _ in range(120)]}}

def test_frozen_trace_cannot_prove_dynamics():
    assert assess_trace_motion(traces('frozen'))['motion_observed'] is False

def test_motion_then_quiet_tail_is_detected():
    result=assess_trace_motion(traces('settling'))
    assert result['motion_observed'] is True and result['tail_motion_bounded'] is True

def test_continued_drift_fails_tail_motion_check():
    assert assess_trace_motion(traces('drifting'))['tail_motion_bounded'] is False

def test_long_run_accepts_full_601_samples():
    t=traces('settling')
    t['body']['position_traj'] += [[0.,0.,0.]]*481
    t['body']['orientation_traj'] += [[1.,0.,0.,0.]]*481
    result=assess_trace_motion(t,expected_samples=601,tail_samples=60)
    assert result['motion_observed'] and result['tail_motion_bounded']
    assert result['tail_samples']==60

def test_long_run_rejects_short_record():
    with pytest.raises(ValueError):
        assess_trace_motion(traces('settling'),expected_samples=601,tail_samples=60)

from acceptance import validate_timing,validate_run_config

def test_actual_timing_counters_agree():
    assert validate_timing(600,600,10.,[1/60]*600)['simulation_seconds']==pytest.approx(10.)

@pytest.mark.parametrize('observed,elapsed,dts',[(599,10.,[1/60]*600),(600,0.,[1/60]*600),(600,10.,[1/60]*120)])
def test_missing_physics_steps_cannot_pass(observed,elapsed,dts):
    with pytest.raises(ValueError): validate_timing(600,observed,elapsed,dts)

def test_resource_budget_is_bounded():
    cfg={'physics_steps':600,'max_turns':6,'max_simulations':1}
    assert validate_run_config(cfg)==cfg
    for key,value in [('physics_steps',60000),('max_turns',100),('max_simulations',0)]:
        with pytest.raises(ValueError): validate_run_config({**cfg,key:value})

from acceptance import trajectory_limits

def test_transient_escape_is_not_hidden_by_return_to_start():
    trace=traces('frozen')['body']
    trace['position_traj'][60]=[.1,0.,0.]
    trace['initial_position']=[0.,0.,0.];trace['initial_orientation']=[1.,0.,0.,0.]
    assert trajectory_limits(trace,[0.,0.,0.])['passed'] is False

def test_tail_translation_uses_body_center_not_world_origin():
    import numpy as np
    from trimesh.transformations import quaternion_matrix
    center=np.array([3.,3.,0.]);quats=[];positions=[]
    for i in range(120):
        a=np.deg2rad(i*.02)/2;q=np.array([np.cos(a),0.,0.,np.sin(a)])
        quats.append(q);positions.append(center-quaternion_matrix(q)[:3,:3]@center)
    t={'body':{'position_traj':positions,'orientation_traj':quats}}
    result=assess_trace_motion(t,reference_centers={'body':center})
    assert result['tail_motion_bounded']
    assert result['per_body']['body']['max_tail_translation_step_m']<1e-10
