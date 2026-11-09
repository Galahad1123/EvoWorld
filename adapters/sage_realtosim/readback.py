"""Independent CPU acceptance of integrated SAGE scene, actual robot and all dynamic bodies."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from specs import validate_specs,pose_matrix,RobotAdapter
from build import dependencies

def collision_corners(asset):
    stage=Usd.Stage.Open(str(asset));root=stage.GetDefaultPrim();cache=UsdGeom.XformCache();corners=[]
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            if not prim.IsA(UsdGeom.Cube):raise ValueError('Benchmark support checker needs explicit box proxy')
            size=UsdGeom.Cube(prim).GetSizeAttr().Get();T=cache.GetLocalToWorldTransform(prim)*cache.GetLocalToWorldTransform(root).GetInverse()
            corners.extend([list(T.Transform(Gf.Vec3d(x*size,y*size,z*size))) for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
    if not corners:raise ValueError('No collision support geometry')
    return np.array(corners)

def verify(run):
    run=Path(run);inp=json.loads((run/'inputs.json').read_text());scene,robot,task=inp['scene'],inp['robot'],inp['task'];validate_specs(scene,robot,task)
    m=json.loads((run/'runtime.json').read_text());reset=json.loads((run/'reset.json').read_text());rows=json.loads((run/'trace.json').read_text());done=json.loads((run/'completed.json').read_text());bundle=Path(inp['bundle'])
    assert done['passed_execution'] and done['exit_code']==0 and done['rows']==160 and len(rows)==160
    assert m['source_environment_class'].endswith('FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser') and m['fixed_base'] and m['action_body']=='panda_hand' and not m['action_relative']
    adapter=RobotAdapter();adapter.check_joints(m['joint_names']);assert m['body_names'][m['hand_index']]=='panda_hand'
    assert m['physics_dt_s']==.01 and m['control_dt_s']==.02
    assert reset['episode_step']==[0] and np.max(np.abs(reset['qd']))==0 and np.max(np.abs(reset['action']))==0
    np.testing.assert_allclose(reset['q'],robot['default_q'],atol=1e-6);np.testing.assert_allclose(pose_matrix(reset['root'][:7]),pose_matrix(robot['base_pose_world']),atol=1e-6)
    manifest=json.loads((bundle/'manifest-sha256.json').read_text());assert hashlib.sha256((bundle/'manifest-sha256.json').read_bytes()).hexdigest()==inp['manifest_sha256']
    assert all((bundle/path).resolve().is_relative_to(bundle.resolve()) and hashlib.sha256((bundle/path).read_bytes()).hexdigest()==h for path,h in manifest.items())
    dep=dependencies(bundle/'scene.usda',bundle)
    assert set(m['asset_properties'])=={o['id'] for o in scene['objects']}
    for o in scene['objects']:
        p=m['asset_properties'][o['id']];assert p['rigid_body']==o['dynamic'] and len(p['collision_paths'])>0
        assert Path(m['asset_paths'][o['id']]).resolve()==(bundle/o['asset']).resolve()
        if o['dynamic']:assert abs(p['mass_kg']-o['physics']['mass_kg'])<1e-6
    assert Path(m['robot_usd']).resolve()==(bundle/robot['asset']).resolve()
    previous=None;wrist_errors=[]
    offset=pose_matrix(m['wrist_offset_ros']['position']+m['wrist_offset_ros']['quaternion_wxyz'])
    for i,row in enumerate(rows):
        assert row['step']==i+1 and row['label']==('settle' if i<100 else 'up' if i<130 else 'return')
        action=adapter.action(row['action'],previous=previous);previous=action
        assert np.isfinite(np.r_[row['q'],row['qd'],row['hand_base'],np.array(row['contact_forces_world']).ravel()]).all()
        limits=np.array(m['joint_limits']);assert np.all(np.array(row['q'])>=limits[:,0]-.001) and np.all(np.array(row['q'])<=limits[:,1]+.001)
        np.testing.assert_allclose(pose_matrix(row['robot_base']),pose_matrix(robot['base_pose_world']),atol=1e-5)
        expected=pose_matrix(row['robot_base'])@pose_matrix(row['hand_base'])@offset
        actual=pose_matrix(row['wrist']['position_world']+row['wrist']['quaternion_ros_wxyz']);error=float(np.max(np.abs(expected-actual)));wrist_errors.append(error);assert error<1e-5
        assert all(abs(t-(i+1)*.02)<1e-4 for t in row['camera_time_s'].values())
    np.testing.assert_allclose(np.diff([r['sim_time_s'] for r in rows]),.02,atol=1e-7)
    free_errors={}
    for label in ['up','return']:
        row=[r for r in rows if r['label']==label][-1];error=float(np.linalg.norm(np.array(row['action'][:3])-row['hand_base'][:3]));assert error<task['acceptance']['endpoint_error_m'];free_errors[label]=error
    actual_up=np.array(rows[129]['hand_base'][:3])-rows[99]['hand_base'][:3];assert actual_up[2]>.024
    assert np.max(np.abs(np.array(rows[129]['q'])-rows[99]['q']))>.01
    forces=np.linalg.norm(np.array([r['contact_forces_world'] for r in rows]),axis=2)
    forbidden=[i for i,n in enumerate(m['contact_body_names']) if n!='panda_link0'];max_force=float(forces[:,forbidden].max());assert max_force<task['acceptance']['unintended_contact_N'],('Unintended robot contact',max_force)
    objects={};table_z=float(np.array(scene['world_T_source'])[2,3])
    for name in scene['dynamic_ids']:
        o=next(o for o in scene['objects'] if o['id']==name);state=np.array([r['objects'][name] for r in rows]);assert state.shape==(160,13) and np.isfinite(state).all()
        np.testing.assert_allclose(reset['objects'][name][:7],o['pose_world'],atol=1e-6)
        drop=float(reset['objects'][name][2]-state[-1,2]);assert drop>.0005,('Dynamic body did not settle under gravity',name,drop)
        corners=collision_corners(bundle/o['asset']);R=Rotation.from_quat(state[:,[4,5,6,3]]).as_matrix();world=np.einsum('nij,kj->nki',R,corners)+state[:,None,:3];bottom=world[:,:,2].min(axis=1)-table_z
        span=float(np.linalg.norm(np.ptp(state[-51:,:3],axis=0)));speed=float(np.linalg.norm(state[-51:,7:10],axis=1).max());assert span<task['acceptance']['object_tail_span_m'] and speed<task['acceptance']['object_tail_speed_m_s']
        assert abs(bottom[-1])<task['acceptance']['support_error_m'] and bottom.min()>-task['acceptance']['support_error_m']
        assert np.max(np.linalg.norm(state[:,:2]-np.array(o['pose_world'][:2]),axis=1))<.002
        objects[name]={'actual_drop_m':drop,'tail_span_m':span,'tail_peak_sampled_speed_m_s':speed,'final_support_error_m':float(bottom[-1]),'minimum_support_error_m':float(bottom.min()),'mass_kg':m['asset_properties'][name]['mass_kg']}
    for tag,step in [('settled',100),('raised',130),('final',160)]:
        image=json.loads((run/(tag+'-images.json')).read_text());assert image['trace_step']==step and abs(image['sim_time_s']-rows[step-1]['sim_time_s'])<1e-7
        with np.load(run/(tag+'-observations.npz')) as f:
            adapter.validate_observation({k:f[k] for k in f.files})
            for name in ['left','right']:assert f['rgb_'+name].std()>5
    return {'passed':True,'control_states':160,'physics_steps':320,'simulated_seconds':3.2,'actual_vertical_motion_m':float(actual_up[2]),'endpoint_error_m':free_errors,'maximum_nonbase_robot_contact_N':max_force,'maximum_wrist_transform_error':max(wrist_errors),'objects':objects,'dependencies':dep,'scope':'cached real-photo reconstruction + original SAGE Franka interface; no grasp/planner/RL certification'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();report=verify(args.run);(Path(args.run)/'readback.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='dependencies'},indent=2))
