"""Independent CPU acceptance of actual simulation trace and sensor/body agreement."""
import json,sys
from pathlib import Path
import numpy as np
from contract import RobotAdapter,TaskAdapter,pose_matrix,transform_pose

def verify(run):
    run=Path(run);done=json.loads((run/'completed.json').read_text());m=json.loads((run/'runtime.json').read_text());rows=json.loads((run/'trace.json').read_text())
    assert done['passed_execution'] and done['exit_code']==0 and len(rows)==270
    if (run/'simulator-exit-code.txt').exists():assert int((run/'simulator-exit-code.txt').read_text())==0
    a=RobotAdapter();a.check_schema(m['schema']);a.check_joints(m['joint_names']);assert m['fixed_base'] and m['body_names'][m['hand_index']]=='panda_hand'
    assert m['control_dt']==.02 and m['physics_dt']==.01
    for snap in m['reset_states']:
        np.testing.assert_allclose(snap['q'][0],m['default_q'],atol=1e-7);assert snap['episode_step']==[0];assert np.max(np.abs(snap['root'][0][7:]))==0;assert np.max(np.abs(snap['action']))==0
    assert m['reset_states'][0]==m['reset_states'][1]
    previous=None
    for i,r in enumerate(rows):
        if i==75:previous=None
        a.action(r['action'],previous=previous);previous=r['action']
        assert np.isfinite(np.r_[r['q'],r['qd'],r['hand_base'],np.array(r['finger_force_world']).ravel()]).all()
        limits=np.array(m['joint_limits']);assert np.all(np.array(r['q'])>=limits[:,0]-.001) and np.all(np.array(r['q'])<=limits[:,1]+.001)
        assert all(abs(t-r['step']*.02)<1e-4 for t in r['camera_time_s'].values())
    times=np.array([r['sim_time_s'] for r in rows]);np.testing.assert_allclose(np.diff(times),.02,atol=1e-7)
    for key in ['q','hand_base']:
        x=np.array([r[key] for r in rows if r['repeat']==0]);y=np.array([r[key] for r in rows if r['repeat']==1]);np.testing.assert_allclose(x,y,atol=1e-6,rtol=0)
    errors={}
    for repeat in (0,1):
        for label in ('free_up','free_return'):
            rs=[r for r in rows if r['repeat']==repeat and r['label']==label];assert len(rs)==30
            err=float(np.linalg.norm(np.array(rs[-1]['action'][:3])-rs[-1]['hand_base'][:3]));assert err<.005
            errors[f'{repeat}_{label}']=err
        s=[r for r in rows if r['repeat']==repeat and r['label']=='settle'][-1];u=[r for r in rows if r['repeat']==repeat and r['label']=='free_up'][-1];assert u['hand_base'][2]-s['hand_base'][2]>.024
    def tail(label):return [r for r in rows if r['label']==label][-5:]
    empty=float(np.mean([r['q'][-2:] for r in tail('empty_close')]));blocked=float(np.mean([r['q'][-2:] for r in tail('contact_close')]));opened=float(np.mean([r['q'][-2:] for r in tail('contact_open')]))
    forces=np.array([r['finger_force_world'] for r in tail('contact_close')]);force=float(np.linalg.norm(forces,axis=2).min());assert TaskAdapter().contact_success(blocked,empty,force,opened)
    # A camera attached to hand must actually follow it; catches stale USD/Fabric poses.
    offset=pose_matrix(m['wrist_offset_ros']['position']+m['wrist_offset_ros']['quaternion_wxyz'])
    expected=[pose_matrix(m['base_pose'])@pose_matrix(row['hand_base'])@offset for row in rows]
    attachment_errors=[float(np.linalg.norm(T[:3,3]-row['wrist_position_world'])) for T,row in zip(expected,rows)]
    assert max(attachment_errors)<.0001,('Wrist camera attachment drift',max(attachment_errors))
    camera=m['cameras']['wrist'];actual=pose_matrix(camera['position_world']+camera['quaternion_ros_wxyz'])
    error=float(np.linalg.norm(expected[-1][:3,3]-actual[:3,3]));assert error<.0001,('Wrist camera not attached to live hand',error)
    rotation_error=float(np.max(np.abs(expected[-1][:3,:3]-actual[:3,:3])));assert rotation_error<1e-5,('Wrist camera mount rotation',rotation_error)
    with np.load(run/'observations-final.npz') as ob:
        schema=a.validate_observation({k:ob[k] for k in ob.files})
        for name in ('left','right'):assert ob['rgb_'+name].std()>5,('Uninformative RGB',name)
    with np.load(run/'observations-free.npz') as ob:
        for name in ('left','right','wrist'):assert ob['rgb_'+name].std()>5,('Uninformative free-space RGB',name)
    for c in m['cameras'].values():
        np.testing.assert_allclose(c['intrinsic_matrix'],[[128*10/20.955,0,64],[0,128*10/20.955,64],[0,0,1]],atol=1e-4)
    return {'passed':True,'states':len(rows),'physics_steps':540,'simulated_seconds':5.4,'repeat_joint_max_difference':float(np.max(np.abs(np.array([r['q'] for r in rows[:75]])-np.array([r['q'] for r in rows[75:150]])))),'endpoint_errors_m':errors,'empty_finger_position_m':empty,'blocked_finger_position_m':blocked,'contact_force_min_last5_perfinger_N':force,'reopened_finger_position_m':opened,'wrist_attachment_error_m':error,'wrist_attachment_max_all_steps_m':max(attachment_errors),'wrist_rotation_matrix_max_error':rotation_error,'observation':schema,'step_wall_seconds':{'median':float(np.median([r['wall_step_s'] for r in rows])),'p95':float(np.percentile([r['wall_step_s'] for r in rows],95))},'scope':'Original Franka actions/observations in diagnostic fixture; no pick-place, physical robot or photo scene integration'}
if __name__=='__main__':
    r=verify(sys.argv[1]);(Path(sys.argv[1])/'readback.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
