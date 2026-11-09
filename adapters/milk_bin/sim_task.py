"""Isaac Lab task adapter preserving SAGE assets, controller and camera poses."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
from evoworld.config import resource
from evoworld.runtime import isaac_experience

def create_task(seed=710,output=None):
 import os,sys
 for package in ['omni.isaac.lab','omni.isaac.lab_assets','omni.isaac.lab_tasks']:
  sys.path.insert(0,str(resource('expert_source')/'IsaacLab/source/extensions'/package))
 os.environ['EVOWORLD_SAGE_ASSET_ROOT']=str(resource('isaac_assets'))
 from omni.isaac.lab.app import AppLauncher
 app=AppLauncher(headless=True,enable_cameras=True,experience=isaac_experience()).app
 from omni.isaac.lab.envs import ManagerBasedRLEnv
 from omni.isaac.lab_tasks.manager_based.manipulation.lift_single_obj_scene.config.franka.ik_abs_env_cfg import FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser
 import milk_runtime as milk
 import torch
 template=resource('env_template')
 import yaml,tempfile
 data=yaml.safe_load(template.read_text())
 for key in ('scene_save_dir','usd_collection_dir'):
  if not Path(data[key]).is_absolute():data[key]=str((template.parent/data[key]).resolve())
 with tempfile.TemporaryDirectory(prefix='evoworld-env-') as td:
  resolved=Path(td)/'env_init.yaml';resolved.write_text(yaml.safe_dump(data))
  cfg=FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser(config_yaml=str(resolved))
 milk.configure(cfg,seed)
 cfg.scene.num_envs=1;cfg.sim.device='cuda:0';cfg.terminations.time_out=None
 # Policy observations use RGB only; expert M2T2 collection keeps its depth cameras.
 for name in ['camera_tabletop','camera_wrist']:getattr(cfg.scene,name).data_types=['rgb']
 for key in vars(cfg.scene):
  if key.startswith('camera_') and key not in ['camera_tabletop','camera_wrist']:setattr(cfg.scene,key,None)
 for key in vars(cfg.observations.policy):
  if hasattr(getattr(cfg.observations.policy,key),'func') and key not in ['rgb_tabletop','rgb_wrist']:setattr(cfg.observations.policy,key,None)
 if output:
  Path(output).mkdir(parents=True,exist_ok=True);os.chdir(output)
 env=ManagerBasedRLEnv(cfg=cfg)
 milk.photo.attach(env)
 profile=milk.SETTINGS['tabletop_camera']
 env.scene['camera_tabletop'].set_world_poses_from_view(torch.tensor([profile['position_world_m']],device=env.device),torch.tensor([profile['target_world_m']],device=env.device))
 return MilkBinTask(env,seed),app

class MilkBinTask:
 def __init__(self,env,seed):
  import numpy as np
  from collections import deque
  from pxr import Usd
  from collision_bridge import collect_colliders
  import milk_runtime as milk
  self.env=env;self.device=env.device;self.rng=np.random.default_rng(seed)
  self.roots=deque(maxlen=51);self.fingers=deque(maxlen=51);self.gripper_commands=deque(maxlen=51);self.done=False;self.steps=0
  self.vertices,_,_=collect_colliders(Usd.Stage.Open(str(milk.BUNDLE/milk.photo.lookup['milk']['asset'])),'/Asset',True)

 def _tensor(self,x):
  import torch
  return torch.as_tensor(x,device=self.device,dtype=torch.float32)

 def _pose(self):
  import torch
  ee=self.env.scene['ee_frame'].data
  return torch.cat([ee.target_pos_source.reshape(1,3),ee.target_quat_source.reshape(1,4)],dim=-1)

 def _obs(self):
  import torch
  q=self.env.scene['robot'].data.joint_pos
  state=torch.cat([q[:,:7],q[:,7:9].sum(-1,keepdim=True)],dim=-1)
  return {'policy':{'rgb_tabletop':self.env.scene['camera_tabletop'].data.output['rgb'].clone(),
                   'rgb_wrist':self.env.scene['camera_wrist'].data.output['rgb'].clone(),
                   'state':state.clone()}}

 def reset(self,seed=None,env_ids=None):
  import numpy as np,torch
  import milk_runtime as milk
  from scipy.spatial.transform import Rotation
  if env_ids is not None:assert len(env_ids)==1 and int(env_ids[0])==0
  if seed is not None:self.rng=np.random.default_rng(seed)
  env=self.env;env.reset(seed=seed);robot=env.scene['robot']
  root=torch.zeros((1,13),device=self.device);root[:,:3]=self._tensor([[2.08,1.49,.8017813]]);root[:,3:7]=self._tensor([[.70710678,0,0,.70710678]])
  robot.write_root_link_state_to_sim(root);robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),torch.zeros_like(robot.data.default_joint_vel));robot.reset()
  for obj in milk.SPEC['objects']:
   if not obj['dynamic']:continue
   state=np.zeros((1,13));state[0,:7]=obj['pose_world']
   if obj['id']=='milk':
    state[0,:2]+=self.rng.uniform(-.04,.04,2)
    state[0,3:7]=Rotation.from_euler('z',self.rng.uniform(-20,20),degrees=True).as_quat(scalar_first=True)
   env.scene[obj['id']].write_root_state_to_sim(self._tensor(state));env.scene[obj['id']].reset()
  env.action_manager.reset();env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
  hold=torch.cat([self._pose(),torch.ones((1,1),device=self.device)],dim=-1)
  for _ in range(21):env.step(hold)
  self.roots.clear();self.fingers.clear();self.gripper_commands.clear();self.done=False;self.steps=0;self.last_success=False
  return self._obs(),{}

 def step(self,action):
  import numpy as np,torch
  import milk_runtime as milk
  from actions import delta_to_absolute
  from task_success import check_window
  if self.done:
   return self._obs(),torch.zeros(1,device=self.device),torch.tensor([self.last_success],device=self.device),torch.tensor([not self.last_success],device=self.device),{}
  value=action.detach().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
  assert value.shape==(1,7) and np.isfinite(value).all()
  absolute=delta_to_absolute(milk.old.cpu(self._pose()),value)
  self.env.step(self._tensor(absolute));self.steps+=1
  root=milk.old.cpu(self.env.scene['milk'].data.root_state_w[0]).copy()
  self.gripper_commands.append(float(absolute[0,-1]))
  self.roots.append(root);self.fingers.append(milk.old.cpu(self.env.scene['robot'].data.joint_pos[0,-2:]).copy())
  result=check_window(self.roots,self.fingers,self.vertices,milk.BOX_POS,milk.BOX_INNER,milk.FLOOR,self.env.step_dt,gripper_commands=self.gripper_commands)
  success=result['success'];failed=not np.isfinite(root).all() or root[2]<milk.BOX_POS[2]-.1
  timeout=self.steps>=650;self.done=success or failed or timeout;self.last_success=success
  reward=1.0-.001 if success else -.001
  return self._obs(),self._tensor([reward]),torch.tensor([success],device=self.device),torch.tensor([failed or timeout],device=self.device),{'physical_acceptance':result}

 def close(self):self.env.close()
