"""Opt-in milk/box task over the accepted SAGE photo scene and office camera."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'adapters/sage_photo_dp'))
import photo_runtime as photo
spec=importlib.util.spec_from_file_location('milk_office',ROOT/'adapters/sage_office/runtime.py')
office=importlib.util.module_from_spec(spec);spec.loader.exec_module(office)
from evoworld.config import resource
BOX_ASSET=resource('storage_box')
BOX_POS=np.array([1.93,1.85,.8007813188675777])
BOX_INNER=[.0876,.0926];FLOOR=.0015
SETTINGS=None;ENV=None
box={'id':'storage_box','asset':str(BOX_ASSET),'dynamic':False,'pose_world':BOX_POS.tolist()+[1,0,0,0],'size_m':[.09,.095,.055]}
assert not any(o['id']=='storage_box' for o in photo.SPEC['objects'])
photo.SPEC['objects'].append(box)
SPEC=photo.SPEC;BUNDLE=photo.BUNDLE;old=photo.old
transforms=photo.transforms;meshes=photo.meshes;robot_location=photo.robot_location

def configure(cfg,seed):
 global SETTINGS
 photo.configure(cfg,seed)
 selected=json.loads((ROOT/'configs/selected-office.json').read_text())
 SETTINGS=office.configure(cfg,resource('office_background'),ROOT/selected['tabletop_camera_profile'])
 from evoworld.observations import apply_wrist
 SETTINGS['wrist_camera']=apply_wrist(cfg.scene.camera_wrist)
 for name in ['camera_tabletop','camera_wrist']:
  cam=getattr(cfg.scene,name);cam.width=224;cam.height=224
 return cfg

def attach(env):
 global ENV
 ENV=env;photo.attach(env);office.attach(env,SETTINGS)
 p=Path('scene-profile.json');d=json.loads(p.read_text());d.update(task='milk_in_storage_box',box={'position':BOX_POS.tolist(),'inner_xy':BOX_INNER,'floor_local_z':FLOOR},step_dt=env.step_dt,office=SETTINGS,observation_contract=SETTINGS['wrist_camera']['observation_contract']);p.write_text(json.dumps(d,indent=2))

def randomize(transforms):
 from scipy.spatial.transform import Rotation
 nominal=np.array(photo.lookup['milk']['pose_world'][:3]);nominal[:2]+=np.random.uniform(-.04,.04,2)
 transforms['milk']['position']=nominal
 transforms['milk']['rotation']=Rotation.from_euler('z',np.random.uniform(-20,20),degrees=True).as_quat(scalar_first=True)

def snapshot(*args):
 # Original collector adjusts old cameras every reset; reassert approved camera.
 office.attach(args[0],SETTINGS);photo.snapshot(*args)

def trace(*args):photo.trace(*args)

def install(namespace):
 photo.install(namespace)
 namespace['get_place_location']=lambda *args:BOX_POS+np.array([0,0,FLOOR])
 original=namespace['curobo_plan_traj']
 def plan(*args,**kwargs):
  from dwell import prepend_dwell
  trajectory,success=original(*args,**kwargs)
  ee=ENV.scene['ee_frame'].data
  pose=torch.cat([ee.target_pos_source[0].reshape(-1),ee.target_quat_source[0].reshape(-1)])
  output=prepend_dwell(old.cpu(trajectory),old.cpu(pose),ENV.step_dt,.3)
  return torch.as_tensor(output,device=trajectory.device,dtype=trajectory.dtype),success
 namespace['curobo_plan_traj']=plan

def dwell_steps():return int(np.ceil(.3/ENV.step_dt-1e-9))

def acceptance_result(attempt,end):
 from task_success import check_window
 from collision_bridge import collect_colliders
 from pxr import Usd
 rows=[json.loads(x) for x in Path('physical-trace.jsonl').read_text().splitlines()]
 roots=[json.loads(x) for x in Path('root-trace.jsonl').read_text().splitlines()]
 rows=[r for r in rows if r['attempt']==attempt and 21<=r['iteration']<end]
 roots=[r for r in roots if r['attempt']==attempt and 21<=r['iteration']<end]
 assert len(rows)==len(roots)==end-21
 assert [r['iteration'] for r in rows]==[r['iteration'] for r in roots]
 milk=photo.lookup['milk'];v,_,_=collect_colliders(Usd.Stage.Open(str(BUNDLE/milk['asset'])),'/Asset',True)
 result=check_window([r['root'] for r in roots],[r['q'][-2:] for r in rows],v,BOX_POS,BOX_INNER,FLOOR,json.loads(Path('scene-profile.json').read_text())['step_dt'],gripper_commands=[r['action'][-1] for r in rows])
 result.update(attempt=attempt,end_exclusive=end)
 return result

def accept_demo(attempt,end):
 result=acceptance_result(attempt,end)
 with Path('physical-acceptance.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
 print('MILK_BOX_ACCEPTANCE '+json.dumps(result),flush=True)
 return result['success']

def capture_before_step(env,attempt,iteration):
 ee=env.scene['ee_frame'].data
 record={'attempt':attempt,'iteration':iteration,
         'ee_base':old.cpu(torch.cat([ee.target_pos_source[0].reshape(-1),ee.target_quat_source[0].reshape(-1)])).tolist(),
         'joint_pos':old.cpu(env.scene['robot'].data.joint_pos[0]).tolist()}
 with Path('pre-action-state.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
