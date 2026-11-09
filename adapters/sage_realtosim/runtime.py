"""Load portable reconstruction through original SAGE Franka IsaacLab, then bounded smoke."""
import argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True);parser.add_argument('--output',required=True)
from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();app=AppLauncher(args).app
import hashlib,json,traceback,time
import numpy as np
import torch
from PIL import Image
from specs import validate_specs,pose_matrix,transform_pose,RobotAdapter,action_record
from omni.isaac.lab_tasks.manager_based.manipulation.lift_single_obj_scene.config.franka.ik_abs_env_cfg import FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.assets import RigidObjectCfg,AssetBaseCfg
from omni.isaac.lab.sensors import CameraCfg,ContactSensorCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
import omni.isaac.lab.sim as sim_utils
from pxr import UsdPhysics,UsdGeom,Usd,UsdUtils

bundle=Path(args.bundle).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
status=1;rows=[];started=time.monotonic();env=None

def cpu(x):return x.detach().cpu().numpy()
def write(name,value):(out/name).write_text(json.dumps(value,indent=2))
def camera_record(camera):
    return {'position_world':cpu(camera.data.pos_w[0]).tolist(),'quaternion_ros_wxyz':cpu(camera.data.quat_w_ros[0]).tolist(),'intrinsic_matrix':cpu(camera.data.intrinsic_matrices[0]).tolist(),'resolution_hw':list(camera.image_shape)}
try:
    scene=json.loads((bundle/'scene_spec.json').read_text());robot_spec=json.loads((bundle/'robot_spec.json').read_text());task=json.loads((bundle/'task_spec.json').read_text());validate_specs(scene,robot_spec,task)
    manifest=json.loads((bundle/'manifest-sha256.json').read_text())
    for path,h in manifest.items():
        p=(bundle/path).resolve();assert p.is_relative_to(bundle) and hashlib.sha256(p.read_bytes()).hexdigest()==h,('Assethash mismatch',path)
    write('inputs.json',{'bundle':str(bundle),'manifest_sha256':hashlib.sha256((bundle/'manifest-sha256.json').read_bytes()).hexdigest(),'scene':scene,'robot':robot_spec,'task':task})
    # Original SAGE environment constructor accepts its usual config; assets are inserted below
    # as native USD references instead of invoking generation or converting existing assets.
    empty=out/'sage-config';empty.mkdir();(empty/'rigid_object_property_dict.json').write_text('{}')
    cfg_file=out/'sage-config.json';cfg_file.write_text(json.dumps({'scene_save_dir':str(empty),'usd_collection_dir':str(empty),'base_pos':robot_spec['base_pose_world'][:3],'mass_dict':{},'room_dict':{'benchmark':{'x_min':-.6,'x_max':.6,'y_min':-.6,'y_max':.3,'height':1.5}}}))
    cfg=FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser(config_yaml=str(cfg_file));cfg.scene.num_envs=1
    for name,value in list(vars(cfg.scene).items()):
        if 'camera_' in name and name not in {'camera_left','camera_right','camera_wrist'}:setattr(cfg.scene,name,None)
    for name in list(vars(cfg.observations.policy)):
        if '_vis' in name:setattr(cfg.observations.policy,name,None)
    cfg.scene.robot.spawn.usd_path=str(bundle/robot_spec['asset']);cfg.scene.robot.init_state.pos=tuple(robot_spec['base_pose_world'][:3]);cfg.scene.robot.init_state.rot=tuple(robot_spec['base_pose_world'][3:]);cfg.scene.robot.init_state.joint_pos=dict(zip(robot_spec['joint_names'],robot_spec['default_q']))
    cfg.scene.robot.spawn.activate_contact_sensors=True;cfg.scene.ee_frame.debug_vis=False
    for o in scene['objects']:
        cls=RigidObjectCfg if o['dynamic'] else AssetBaseCfg
        entry=cls(prim_path='{ENV_REGEX_NS}/'+o['id'],spawn=sim_utils.UsdFileCfg(usd_path=str(bundle/o['asset'])),init_state=cls.InitialStateCfg(pos=tuple(o['pose_world'][:3]),rot=tuple(o['pose_world'][3:])))
        setattr(cfg.scene,o['id'],entry)
    # A floor is a designed visualization/support reference, not reconstructed desk legs.
    cfg.scene.ground=AssetBaseCfg(prim_path='/World/Ground',spawn=sim_utils.GroundPlaneCfg())
    cfg.scene.light.spawn.intensity=700
    for name in ['overview','detail']:
        w,h=task['cameras'][name]['resolution_wh']
        setattr(cfg.scene,'camera_'+name,CameraCfg(prim_path='{ENV_REGEX_NS}/camera_'+name,update_period=0,height=h,width=w,data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=24,horizontal_aperture=36,clipping_range=(.01,100))))
    wrist_offset=cfg.scene.camera_wrist.offset;cfg.scene.camera_wrist.prim_path='{ENV_REGEX_NS}/camera_wrist'
    cfg.scene.robot_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/panda_.*',update_period=0,history_length=1)
    cfg.seed=task['seed'];cfg.terminations.time_out=None;cfg.observations.policy.concatenate_terms=False;cfg.observations.policy.enable_corruption=False;cfg.sim.device='cuda:0';cfg.sim.dt=task['physics_dt_s'];cfg.decimation=2;cfg.sim.use_fabric=True
    env=ManagerBasedRLEnv(cfg=cfg);robot=env.scene['robot'];adapter=RobotAdapter(env);hand=robot.body_names.index('panda_hand')
    render_original=env.sim.render
    def render_with_wrist(*a,**kw):
        link=cpu(robot.root_physx_view.get_link_transforms()[0,hand]);hand_w=np.r_[link[:3],link[6],link[3:6]];optical=transform_pose(list(wrist_offset.pos)+list(wrist_offset.rot),pose_matrix(hand_w))
        env.scene['camera_wrist'].set_world_poses(torch.tensor([optical[:3]],device=env.device,dtype=torch.float32),torch.tensor([optical[3:]],device=env.device,dtype=torch.float32),convention='ros')
        return render_original(*a,**kw)
    env.sim.render=render_with_wrist
    adapter.reset(seed=task['seed'])
    for name,cam in task['cameras'].items():
        sensor=env.scene['camera_'+name];sensor.set_world_poses_from_view(torch.tensor([cam['eye']],device=env.device),torch.tensor([cam['target']],device=env.device));sensor.reset()
    env.scene['camera_wrist'].reset()
    def hand_pose():
        p,q=subtract_frame_transforms(robot.data.root_state_w[:,:3],robot.data.root_state_w[:,3:7],robot.data.body_state_w[:,hand,:3],robot.data.body_state_w[:,hand,3:7]);return np.r_[cpu(p[0]),cpu(q[0])]
    write('reset.json',{'q':cpu(robot.data.joint_pos[0]).tolist(),'qd':cpu(robot.data.joint_vel[0]).tolist(),'root':cpu(robot.data.root_state_w[0]).tolist(),'objects':{n:cpu(env.scene[n].data.root_state_w[0]).tolist() for n in scene['dynamic_ids']},'episode_step':cpu(env.episode_length_buf).tolist(),'action':cpu(env.action_manager.action).tolist()})
    def snapshot(tag,policy):
        frames={};poses={}
        for name in ['left','right','wrist','overview','detail']:
            cam=env.scene['camera_'+name];pixels=cpu(cam.data.output['rgb'][0]);Image.fromarray(pixels).save(out/(tag+'-'+name+'.png'));poses[name]=camera_record(cam);frames[name]=int(cpu(cam.frame)[0])
        write(tag+'-images.json',{'meaning':'live render read after actual control step; no offline pose substitution','trace_step':len(rows),'sim_time_s':float(env.sim.current_time),'camera_software_clock_s':{n:float(cpu(env.scene['camera_'+n]._timestamp)[0]) for n in ['left','right','wrist','overview','detail']},'frames':frames,'cameras':poses})
        np.savez_compressed(out/(tag+'-observations.npz'),**policy)
    def step(a,label):
        begin=time.monotonic();obs,reward,terminated,truncated,info=adapter.step(a);policy={k:cpu(v[0]) for k,v in obs['policy'].items()};adapter.validate_observation(policy)
        rows.append({'step':len(rows)+1,'label':label,'sim_time_s':float(env.sim.current_time),'action':action_record(a),'hand_base':hand_pose().tolist(),'robot_base':cpu(robot.data.root_state_w[0,:7]).tolist(),'q':cpu(robot.data.joint_pos[0]).tolist(),'qd':cpu(robot.data.joint_vel[0]).tolist(),'objects':{n:cpu(env.scene[n].data.root_state_w[0]).tolist() for n in scene['dynamic_ids']},'contact_forces_world':cpu(env.scene['robot_contacts'].data.net_forces_w[0]).tolist(),'wrist':camera_record(env.scene['camera_wrist']),'camera_time_s':{n:float(cpu(env.scene['camera_'+n]._timestamp)[0]) for n in ['left','right','wrist']},'wall_step_s':time.monotonic()-begin})
        return policy
    home=hand_pose()
    for i in range(task['settle_steps']):policy=step(np.r_[home,1],'settle')
    snapshot('settled',policy);home=hand_pose();offset=np.array(task['motion_offset_base_m']);goal=home.copy();goal[:3]+=offset
    for i in range(task['motion_steps_each']):
        a=home.copy();a[:3]+=offset*min((i+1)/15,1);policy=step(np.r_[a,1],'up')
    snapshot('raised',policy)
    for i in range(task['motion_steps_each']):
        a=goal.copy();a[:3]-=offset*min((i+1)/15,1);policy=step(np.r_[a,1],'return')
    snapshot('final',policy)
    stage=env.sim.stage
    properties={}
    for o in scene['objects']:
        root=stage.GetPrimAtPath('/World/envs/env_0/'+o['id']);assert root
        properties[o['id']]={'rigid_body':root.HasAPI(UsdPhysics.RigidBodyAPI),'mass_kg':float(UsdPhysics.MassAPI(root).GetMassAttr().Get()) if o['dynamic'] else None,'collision_paths':[str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]}
    write('runtime.json',{'joint_names':robot.joint_names,'body_names':robot.body_names,'contact_body_names':env.scene['robot_contacts'].body_names,'fixed_base':robot.is_fixed_base,'hand_index':hand,'joint_limits':cpu(robot.data.soft_joint_pos_limits[0]).tolist(),'action_body':cfg.actions.arm_action.body_name,'action_relative':cfg.actions.arm_action.controller.use_relative_mode,'physics_dt_s':env.physics_dt,'control_dt_s':env.step_dt,'observation':adapter.validate_observation(policy),'asset_properties':properties,'robot_usd':cfg.scene.robot.spawn.usd_path,'asset_paths':{o['id']:str(bundle/o['asset']) for o in scene['objects']},'wrist_offset_ros':{'position':list(wrist_offset.pos),'quaternion_wxyz':list(wrist_offset.rot)},'wall_seconds':time.monotonic()-started,'source_environment_class':type(cfg).__module__+'.'+type(cfg).__name__})
    status=0
except Exception:
    (out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
finally:
    try:
        write('trace.json',rows);write('completed.json',{'passed_execution':status==0,'rows':len(rows),'exit_code':status,'wall_seconds':time.monotonic()-started})
    finally:
        app.close(exit_code=status)
