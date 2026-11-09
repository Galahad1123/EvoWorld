"""Bounded real Isaac audit: original SAGE robot/actions/observations in a diagnostic fixture."""
import argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--config',required=True)
from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();app=AppLauncher(args).app
import json,time,traceback,os
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from PIL import Image
from contract import RobotAdapter,TaskAdapter,pose_matrix,transform_pose
from omni.isaac.lab_tasks.manager_based.manipulation.lift_single_obj_scene.config.franka.ik_abs_env_cfg import FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.assets import RigidObjectCfg,AssetBaseCfg
from omni.isaac.lab.sensors import ContactSensorCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.math import subtract_frame_transforms
out=Path(args.output);out.mkdir(parents=True,exist_ok=True);status=1
rows=[];adapter=RobotAdapter();task=TaskAdapter();started=time.monotonic()
def cpu(x):return x.detach().cpu().numpy()
try:
    cfg=FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser(config_yaml=args.config)
    # Runtime-only diagnostic fixture; preserve existing source/planners and original actions.
    keep={'num_envs','env_spacing','replicate_physics','lazy_sensor_update','filter_collisions','robot','ee_frame','light','camera_left','camera_right','camera_wrist'}
    for name,val in list(vars(cfg.scene).items()):
        if name not in keep and hasattr(val,'prim_path'):setattr(cfg.scene,name,None)
    for name in list(vars(cfg.observations.policy)):
        if '_vis' in name:setattr(cfg.observations.policy,name,None)
    cfg.scene.num_envs=1;cfg.scene.robot.init_state.pos=(0,0,0);cfg.scene.robot.init_state.rot=(1,0,0,0)
    cfg.scene.robot.spawn.activate_contact_sensors=True;cfg.scene.ee_frame.debug_vis=False
    cfg.scene.ground=AssetBaseCfg(prim_path='/World/Ground',spawn=sim_utils.GroundPlaneCfg(),init_state=AssetBaseCfg.InitialStateCfg(pos=(0,0,-.08)))
    cfg.scene.contact_block=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/ContactBlock',spawn=sim_utils.CuboidCfg(size=(.045,.05,.045),rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,disable_gravity=True),collision_props=sim_utils.CollisionPropertiesCfg(),mass_props=sim_utils.MassPropertiesCfg(mass=.08),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.1,.35,.8))),init_state=RigidObjectCfg.InitialStateCfg(pos=(1.5,0,.5)))
    # Isaac6 legacy nested camera USD pose remains at authored articulation pose.
    # Keep the original optical mount, but synchronize a world-level camera from live PhysX.
    wrist_offset=cfg.scene.camera_wrist.offset
    cfg.scene.camera_wrist.prim_path='{ENV_REGEX_NS}/camera_wrist'
    cfg.scene.finger_contacts=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/panda_.*finger',update_period=0,history_length=1)
    cfg.terminations.time_out=None;cfg.observations.policy.concatenate_terms=False;cfg.observations.policy.enable_corruption=False;cfg.seed=31;cfg.sim.device='cuda:0';cfg.sim.use_fabric=True
    env=ManagerBasedRLEnv(cfg=cfg);robot=env.scene['robot'];block=env.scene['contact_block'];adapter.check_joints(robot.joint_names)
    hand=robot.body_names.index('panda_hand');assert hand==8,('Original observation index mismatch',robot.body_names)
    assert robot.is_fixed_base
    adapter.bind(env);task=TaskAdapter(adapter)
    render_original=env.sim.render
    def render_with_wrist(*args,**kwargs):
        link=cpu(robot.root_physx_view.get_link_transforms()[0,hand])
        hand_w=np.r_[link[:3],link[6],link[3:6]]
        optical=transform_pose(list(wrist_offset.pos)+list(wrist_offset.rot),pose_matrix(hand_w))
        env.scene['camera_wrist'].set_world_poses(torch.tensor([optical[:3]],device=env.device,dtype=torch.float32),torch.tensor([optical[3:]],device=env.device,dtype=torch.float32),convention='ros')
        return render_original(*args,**kwargs)
    env.sim.render=render_with_wrist
    def hand_pose():
        p,q=subtract_frame_transforms(robot.data.root_state_w[:,:3],robot.data.root_state_w[:,3:7],robot.data.body_state_w[:,hand,:3],robot.data.body_state_w[:,hand,3:7]);return np.r_[cpu(p[0]),cpu(q[0])]
    def camera_set():
        for name,eye in [('left',[.8,.65,.65]),('right',[.8,-.65,.65])]:
            env.scene['camera_'+name].set_world_poses_from_view(torch.tensor([eye],device=env.device),torch.tensor([[.4,0,.3]],device=env.device))
        for name in ['left','right','wrist']:env.scene['camera_'+name].reset()
    def reset():
        task.reset(seed=31);camera_set()
        # Reset caches and observations are sampled only after subsequent real control steps.
        return {'q':cpu(robot.data.joint_pos).tolist(),'root':cpu(robot.data.root_state_w).tolist(),'block':cpu(block.data.root_state_w).tolist(),'action':cpu(env.action_manager.action).tolist(),'episode_step':cpu(env.episode_length_buf).tolist()}
    previous=None;episode_t=0
    def step(a,label,repeat):
        global previous,episode_t
        a=adapter.action(a,previous=previous);previous=a.copy();start=time.monotonic()
        obs,reward,terminated,truncated,info=adapter.step(a);wall=time.monotonic()-start;episode_t+=1
        policy={k:cpu(v[0]) for k,v in obs['policy'].items()};adapter.validate_observation(policy)
        row={'repeat':repeat,'label':label,'step':episode_t,'sim_time_s':float(env.sim.current_time),'action':a.tolist(),'q':cpu(robot.data.joint_pos[0]).tolist(),'qd':cpu(robot.data.joint_vel[0]).tolist(),'hand_base':hand_pose().tolist(),'finger_force_world':cpu(env.scene['finger_contacts'].data.net_forces_w[0]).tolist(),'wall_step_s':wall,'camera_time_s':{n:float(cpu(env.scene['camera_'+n]._timestamp)[0]) for n in ['left','right','wrist']},'camera_frame':{n:int(cpu(env.scene['camera_'+n].frame)[0]) for n in ['left','right','wrist']}}
        row['wrist_position_world']=cpu(env.scene['camera_wrist'].data.pos_w[0]).tolist()
        rows.append(row)
        return policy
    resets=[]
    for repeat in range(2):
        previous=None;episode_t=0;resets.append(reset());home=hand_pose()
        for i in range(15):policy=step(np.r_[home,1],'settle',repeat)
        start_pose=hand_pose();goal=start_pose.copy();goal[2]+=.03
        for i in range(30):
            a=start_pose.copy();a[:3]=start_pose[:3]+(goal[:3]-start_pose[:3])*min((i+1)/15,1);policy=step(np.r_[a,1],'free_up',repeat)
        for i in range(30):
            a=goal.copy();a[:3]=goal[:3]+(start_pose[:3]-goal[:3])*min((i+1)/15,1);policy=step(np.r_[a,1],'free_return',repeat)
    np.savez_compressed(out/'observations-free.npz',**policy)
    for name in ['left','right','wrist']:Image.fromarray(policy['rgb_'+name]).save(out/(name+'-free.png'))
    # Same command in empty space versus a known-width kinematic contact fixture.
    hold=hand_pose()
    for i in range(35):policy=step(np.r_[hold,-1],'empty_close',2)
    for i in range(25):policy=step(np.r_[hold,1],'empty_open',2)
    H=pose_matrix(cpu(robot.data.body_state_w[0,hand,:7]));bpose=transform_pose([0,0,.1034,1,0,0,0],H)
    state=torch.zeros((1,13),device=env.device);state[0,:7]=torch.tensor(bpose,device=env.device);block.write_root_state_to_sim(state);block.reset();env.scene.write_data_to_sim()
    for i in range(35):policy=step(np.r_[hold,-1],'contact_close',2)
    for i in range(25):policy=step(np.r_[hold,1],'contact_open',2)
    cameras={}
    for name in ['left','right','wrist']:
        camera=env.scene['camera_'+name]
        cameras[name]={'intrinsic_matrix':cpu(camera.data.intrinsic_matrices[0]).tolist(),'position_world':cpu(camera.data.pos_w[0]).tolist(),'quaternion_ros_wxyz':cpu(camera.data.quat_w_ros[0]).tolist(),'height':128,'width':128,'update_period':camera.cfg.update_period,'clipping_range':list(camera.cfg.spawn.clipping_range)}
        Image.fromarray(policy['rgb_'+name]).save(out/(name+'.png'))
    limits=robot.data.joint_pos_limits if hasattr(robot.data,'joint_pos_limits') else robot.data.soft_joint_pos_limits
    report={'schema':adapter.schema,'joint_names':robot.joint_names,'body_names':robot.body_names,'hand_index':hand,'fixed_base':robot.is_fixed_base,'joint_limits':cpu(limits[0]).tolist(),'default_q':cpu(robot.data.default_joint_pos[0]).tolist(),'base_pose':cpu(robot.data.root_state_w[0,:7]).tolist(),'action_body':'panda_hand','body_offset_m':[0,0,0],'physics_dt':env.physics_dt,'control_dt':env.step_dt,'observation':adapter.validate_observation(policy),'cameras':cameras,'reset_states':resets,'contact_block_pose':bpose,'contact_block_size_m':[.045,.05,.045],'robot_usd':cfg.scene.robot.spawn.usd_path,'wrist_mount_sync':'world camera, live PhysX hand transform before each render; original local mount unchanged','wrist_offset_ros':{'position':list(wrist_offset.pos),'quaternion_wxyz':list(wrist_offset.rot)},'use_fabric':cfg.sim.use_fabric,'scope':'diagnostic fixture with original robot, actions and12 observations; no original planner modification','wall_seconds':time.monotonic()-started}
    (out/'runtime.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'observations-final.npz',**policy)
    status=0
except Exception:
    (out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
finally:
    (out/'trace.json').write_text(json.dumps(rows));(out/'completed.json').write_text(json.dumps({'passed_execution':status==0,'rows':len(rows),'exit_code':status,'wall_seconds':time.monotonic()-started}))
    app.close(exit_code=status)
