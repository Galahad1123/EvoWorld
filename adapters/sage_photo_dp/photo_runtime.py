import copy,importlib.util,json,os
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
from evoworld.config import resource
BUNDLE=resource('photo_scene')
SPEC=json.loads((BUNDLE/'scene_spec.json').read_text())
SHIFT=np.array([2.08,1.97,.0507813188675777])
for obj in SPEC['objects']:obj['pose_world'][:3]=(np.array(obj['pose_world'][:3])+SHIFT).tolist()
lookup={o['id']:o for o in SPEC['objects']}
mod=importlib.util.spec_from_file_location('old_dp_runtime',ROOT/'adapters/sage_dp/runtime.py');old=importlib.util.module_from_spec(mod);mod.loader.exec_module(old)
ENV=None

def configure(cfg,seed):
 from omni.isaac.lab.assets import RigidObjectCfg,AssetBaseCfg
 import omni.isaac.lab.sim as sim
 old.configure(cfg,seed)
 for k in ['room_evoworld_30dd907ee170_table','room_evoworld_30dd907ee170_cube']:setattr(cfg.scene,k,None)
 for o in SPEC['objects']:
  cls=RigidObjectCfg if o['dynamic'] else AssetBaseCfg
  setattr(cfg.scene,o['id'],cls(prim_path='{ENV_REGEX_NS}/'+o['id'],spawn=sim.UsdFileCfg(usd_path=str(BUNDLE/o['asset'])),init_state=cls.InitialStateCfg(pos=tuple(o['pose_world'][:3]),rot=tuple(o['pose_world'][3:]))))
 cfg.scene.camera_tabletop=copy.deepcopy(cfg.scene.camera_left)
 cfg.scene.camera_tabletop.prim_path='{ENV_REGEX_NS}/camera_tabletop'
 cfg.scene.camera_tabletop.width=128;cfg.scene.camera_tabletop.height=128
 cfg.scene.camera_tabletop.spawn.focal_length=32.0
 cfg.observations.policy.rgb_tabletop=copy.deepcopy(cfg.observations.policy.rgb_left)
 cfg.observations.policy.rgb_tabletop.params={'camera_name':'camera_tabletop'}
 return cfg

def attach(env):
 global ENV
 ENV=env;old.attach(env)
 env.scene['camera_tabletop'].set_world_poses_from_view(torch.tensor([[1.55,1.45,1.45]],device=env.device),torch.tensor([[2.0,1.91,.86]],device=env.device))
 Path('scene-profile.json').write_text(json.dumps({'source_bundle':str(BUNDLE),'objects':SPEC['objects'],'shift':SHIFT.tolist(),'room':'layout_evoworld_cube_control','removed':['old table','control cube'],'speed':.8,'policy_cameras':['rgb_tabletop','rgb_wrist']},indent=2))

def transforms():return {o['id']:{'position':np.array(o['pose_world'][:3]),'rotation':np.array(o['pose_world'][3:])} for o in SPEC['objects'] if o['dynamic']}

def robot_location(*args):
 return (torch.tensor([[2.08,1.49,.8017813]],device='cuda'),torch.tensor([[.70710678,0,0,.70710678]],device='cuda'),torch.tensor([[2.08,1.97,.85]],device='cuda'))

def randomize(transforms):
 from scipy.spatial.transform import Rotation
 nominal=np.array(lookup['milk']['pose_world'][:3]);nominal[:2]+=np.random.uniform(-.02,.02,2)
 transforms['milk']['position']=nominal
 transforms['milk']['rotation']=Rotation.from_euler('z',np.random.uniform(-10,10),degrees=True).as_quat(scalar_first=True)

def meshes(*args):
 from usd_visual import mesh_info
 result={}
 for o in SPEC['objects']:
  pose=o['pose_world']
  if o['dynamic'] and ENV is not None:pose=old.cpu(ENV.scene[o['id']].data.root_state_w[0,:7]).tolist()
  result[o['id']]=mesh_info(BUNDLE/o['asset'],pose,Path.cwd()/'m2t2-textures'/(o['id']+'.png'))
 return result

def snapshot(env,attempt,target,place,out):
 from PIL import Image
 for name in ['tabletop','wrist','left_vis']:
  camera=env.scene['camera_'+name]
  pixels=old.cpu(camera.data.output['rgb'][0])
  Image.fromarray(pixels).save(Path(out,f'initial-{attempt:03}-{name}.png'))
 old.snapshot(env,attempt,target,place,out)
 p=Path(out,f'initial-{attempt:03}.json');d=json.loads(p.read_text());cam=env.scene['camera_tabletop']
 d['cameras']['tabletop']={'position':old.cpu(cam.data.pos_w).tolist(),'quaternion':old.cpu(cam.data.quat_w_ros).tolist(),'intrinsics':old.cpu(cam.data.intrinsic_matrices).tolist()};p.write_text(json.dumps(d,indent=2))

def trace(env,attempt,iteration,target,actions,out):
 old.trace(env,attempt,iteration,target,actions,out)
 record={'attempt':attempt,'iteration':iteration,'root':old.cpu(env.scene[target].data.root_state_w[0]).tolist(),
         'ee_pose':old.cpu(env.scene['ee_frame'].data.target_pos_source[0]).tolist()}
 with Path(out,'root-trace.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 if iteration in [100,200,300]:
  from PIL import Image
  for name in ['tabletop','wrist']:
   Image.fromarray(old.cpu(env.scene['camera_'+name].data.output['rgb'][0])).save(Path(out,f'attempt-{attempt:03}-step-{iteration:03}-{name}.png'))

def install(namespace):
 from collision_bridge import bridge_planner_class
 from retime import retime_pose
 namespace['MotionPlanner']=bridge_planner_class(namespace['MotionPlanner'],{o['id']:{'dynamic':o['dynamic']} for o in SPEC['objects']})
 namespace['export_layout_to_mesh_dict_list_tree_search_with_object_id']=meshes
 namespace['sample_robot_location']=robot_location
 namespace['get_place_location']=lambda *args:np.array([1.93,1.85,.8007813188675777])
 namespace['get_grasp_object_height']=lambda *args:.13
 original=namespace['curobo_plan_traj']
 def plan(*args,**kwargs):
  trajectory,success=original(*args,**kwargs)
  if trajectory is None:raise RuntimeError('Original planner returned no trajectory')
  slowed=retime_pose(trajectory.detach().cpu().numpy(),.8)
  return torch.as_tensor(slowed,device=trajectory.device,dtype=trajectory.dtype),success
 namespace['curobo_plan_traj']=plan


def accept_demo(attempt,end):
 from acceptance import check_episode
 result=check_episode(Path.cwd(),attempt,end)
 with Path('physical-acceptance.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
 print('PHOTO_EXPERT_ACCEPTANCE '+json.dumps(result),flush=True)
 return result['success']
