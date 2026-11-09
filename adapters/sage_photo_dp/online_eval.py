"""Independent online evaluation: native photo scene, frozen checkpoint, real physics."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--selection',required=True)
p.add_argument('--output',required=True);p.add_argument('--mode',choices=['expert','policy'],required=True)
from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import copy,hashlib,json,os,pickle,struct,subprocess,time,traceback,select
import h5py
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from scipy.spatial.transform import Rotation
from pxr import Usd
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.manager_based.manipulation.lift_single_obj_scene.config.franka.ik_abs_env_cfg import FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser
import photo_runtime as photo
from collision_bridge import collect_colliders
from eval_metrics import physical_result,validated_action
ROOT=Path(__file__).resolve().parents[2];run=Path(args.run).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
cpu=photo.old.cpu;proc=None;env=None;status=1;results=[];started=time.monotonic()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def recv():
 if not select.select([proc.stdout],[],[],90)[0]:raise TimeoutError('Policy response timeout')
 head=proc.stdout.read(8)
 if len(head)!=8:raise RuntimeError('Policy worker exited')
 n=struct.unpack('!Q',head)[0];return pickle.loads(proc.stdout.read(n))
def request(msg):
 b=pickle.dumps(msg,protocol=5);proc.stdin.write(struct.pack('!Q',len(b))+b);proc.stdin.flush();return recv()
try:
 selection=json.loads((Path(args.selection)/'selection.json').read_text());checkpoint=selection['selected']['checkpoint']
 assert sha(checkpoint)==selection['selected_sha256']
 for name in ['online_eval.py','photo_policy_worker.py','eval_metrics.py','photo_runtime.py','collision_bridge.py']:
  (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
 data=json.loads((run/'dataset-report.json').read_text());profile=json.loads((run/'scene-profile.json').read_text())
 target=next(o for o in profile['objects'] if o['id']=='milk');desk=next(o for o in profile['objects'] if o['id']=='desk')
 corners,_,_=collect_colliders(Usd.Stage.Open(str(Path(profile['source_bundle'])/target['asset'])),'/Asset',True)
 config=next(run.glob('robomimic_data/*/params/env_init.yaml'))
 cfg=FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser(config_yaml=str(config));photo.configure(cfg,408)
 cfg.scene.num_envs=1;cfg.sim.device='cuda:0';cfg.terminations.time_out=None
 for key in list(vars(cfg.scene)):
  if key.startswith('camera_') and key not in {'camera_left','camera_right','camera_wrist','camera_tabletop'}:setattr(cfg.scene,key,None)
 for key in list(vars(cfg.observations.policy)):
  if '_vis' in key:setattr(cfg.observations.policy,key,None)
 os.chdir(out)
 env=ManagerBasedRLEnv(cfg=cfg);photo.attach(env);robot=env.scene['robot'];obj=env.scene['milk']
 tensor=lambda x:torch.as_tensor(x,device=env.device,dtype=torch.float32)
 keys=['ee_pose','joint_qpos','rgb_tabletop','rgb_wrist']
 if args.mode=='policy':
  worker_env=dict(os.environ);worker_env.pop('PYTHONPATH',None)
  proc=subprocess.Popen([str(ROOT/'envs/sage-dp/bin/python'),str(ROOT/'adapters/sage_photo_dp/photo_policy_worker.py'),checkpoint],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(out/'worker.log').open('w'),env=worker_env)
  ready=recv();assert ready['ready'] and set(ready['obs_keys'])==set(keys)
 seeds=list(range(408,418)) if args.mode=='policy' else [408,409]
 (out/'trial-plan.json').write_text(json.dumps({'seeds':seeds,'mode':args.mode,'checkpoint':checkpoint,'sha256':selection['selected_sha256'],
  'randomization':'milk nominalXY +/-0.02m, yaw +/-10deg; independent default_rng(seed)',
  'policy_horizon_steps':750,'hold_steps':50,'control_hz':50,'video_fps':25},indent=2))
 with h5py.File(run/'training.hdf5','r') as f:
  for i,seed in enumerate(seeds):
   demo=data['test'][i] if args.mode=='expert' else data['test'][0]
   meta=data['episodes'][demo];initial=json.loads((Path(meta['source_run'])/f"initial-{meta['attempt']:03}.json").read_text())
   trial=f'trial-{i:02}';env.reset(seed=seed)
   robot.write_root_link_state_to_sim(tensor(initial['robot_root']));robot.write_joint_state_to_sim(tensor(initial['q']),tensor(initial['qd']));robot.reset()
   for name,values in initial['objects'].items():
    env.scene[name].write_root_state_to_sim(tensor(values));env.scene[name].reset()
   env.action_manager.reset();env.action_manager.process_action(tensor(initial['previous_action']))
   if args.mode=='policy':
    rng=np.random.default_rng(seed);pose=np.array(target['pose_world']);pose[:2]+=rng.uniform(-.02,.02,2)
    pose[3:]=Rotation.from_euler('z',rng.uniform(-10,10),degrees=True).as_quat(scalar_first=True)
    obj.write_root_state_to_sim(tensor(np.r_[pose,np.zeros(6)])[None]);obj.reset()
   for name in ['left','right','tabletop']:
    camera=initial['cameras'][name];env.scene['camera_'+name].set_world_poses(tensor(camera['position']),tensor(camera['quaternion']),convention='ros');env.scene['camera_'+name].reset()
   env.scene['camera_wrist'].reset();env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
   if args.mode=='policy':
    # Settle randomized milk, holding the restored robot's current pose. No
    # expert future action is used in policy mode.
    ee=env.scene['ee_frame'].data
    hold=torch.cat([ee.target_pos_source.reshape(1,3),ee.target_quat_source.reshape(1,4),torch.ones((1,1),device=env.device)],dim=1)
    for _ in range(21):env.step(hold)
   for _ in range(12):env.sim.render()
   for name in ['left','right','tabletop','wrist']:env.scene['camera_'+name].update(0.0,force_recompute=True)
   obs=env.observation_manager.compute()['policy']
   np.savez_compressed(out/(trial+'-initial.npz'),**{k:cpu(obs[k][0]) for k in keys},milk_root=cpu(obj.data.root_state_w[0]),robot_root=cpu(robot.data.root_state_w[0]),q=cpu(robot.data.joint_pos[0]))
   image_mae={};state_error={}
   if args.mode=='expert':
    for k in keys:
     diff=np.abs(cpu(obs[k][0]).astype(float)-f[f'data/{demo}/obs/{k}'][0].astype(float))
     if k.startswith('rgb'):image_mae[k]=float(diff.mean())
     else:state_error[k]=float(diff.max())
    assert max(state_error.values())<.001, state_error
    assert max(image_mae.values())<15, image_mae
   if args.mode=='policy':request({'reset':True,'seed':seed})
   z0=float(cpu(obj.data.root_com_state_w[0,2]));goal=np.asarray(initial['place_world'])[0,:2]
   horizon=len(f[f'data/{demo}/actions']) if args.mode=='expert' else 750
   states=[];roots=[];actions=[];raw=[];joints=[];latencies=[];bad=None;invalid_action=None
   writer=imageio.get_writer(str(out/(trial+'.mp4')),fps=25)
   writer.append_data(np.concatenate([cpu(obs['rgb_tabletop'][0]),cpu(obs['rgb_wrist'][0])],axis=1))
   try:
    for t in range(horizon+50):
     if t<horizon:
      begin=time.monotonic()
      a=f[f'data/{demo}/actions'][t] if args.mode=='expert' else np.asarray(request({'obs':{k:cpu(obs[k][0]) for k in keys}})['action'])
      latencies.append(time.monotonic()-begin)
      try:applied=validated_action(a)
      except ValueError:
       bad='invalid_action';invalid_action={'shape':list(a.shape),'value_repr':repr(a)[:512]};break
      raw.append(a.copy());a=applied
     obs=env.step(tensor(a)[None])[0]['policy']
     states.append(cpu(obj.data.root_com_state_w[0]).copy());roots.append(cpu(obj.data.root_state_w[0]).copy());actions.append(a.copy());joints.append(cpu(robot.data.joint_pos[0]).copy())
     if t%2==1:writer.append_data(np.concatenate([cpu(obs['rgb_tabletop'][0]),cpu(obs['rgb_wrist'][0])],axis=1))
   finally:writer.close()
   states=np.array(states);roots=np.array(roots)
   outcome=physical_result(states,roots,z0,goal,corners,desk['pose_world'][2],cpu(robot.data.joint_pos[0,-2:])) if len(states)>=25 else {'success':False,'failed_checks':['incomplete_trace'],'checks':{},'metrics':{}}
   if bad:outcome['success']=False
   result={'trial':trial,'seed':seed,'mode':args.mode,'steps':len(states),'error':bad,'invalid_action':invalid_action,**outcome,
           'initial_fixture':demo,'expert_initial_image_mae':image_mae,'expert_initial_state_error':state_error,
           'policy_call_mean_s':float(np.mean(latencies)) if latencies else None}
   np.savez_compressed(out/(trial+'.npz'),states=states,roots=roots,actions=np.asarray(actions),raw_actions=np.asarray(raw),joints=np.asarray(joints),goal_xy=goal,z0=z0,corners=corners,table_z=desk['pose_world'][2])
   Image.fromarray(np.concatenate([cpu(obs['rgb_tabletop'][0]),cpu(obs['rgb_wrist'][0])],axis=1)).save(out/(trial+'-final.png'))
   results.append(result);(out/(trial+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 final={'complete':True,'mode':args.mode,'episodes':results,'successes':sum(r['success'] for r in results),'total':len(results),
  'checkpoint':checkpoint,'checkpoint_sha256':selection['selected_sha256'],'wall_seconds':time.monotonic()-started,
  'scope':'native photo-tabletop fixed scene; heldout expert replay or10fresh seeded policy starts; no retraining',
  'success_definition':'lift>3cm,XY<5cm,open fingers>2.5cm,support error<1.5cm,tilt<15deg,last0.5s linear<5cm/s angular<0.2rad/s'}
 (out/'report.json').write_text(json.dumps(final,indent=2));status=0
except Exception:
 error=traceback.format_exc();(out/'error.txt').write_text(error);traceback.print_exc()
 (out/'report.json').write_text(json.dumps({'complete':False,'mode':args.mode,'planned_total':10 if args.mode=='policy' else 2,'finished_episodes':results,'infrastructure_error':error,'success_rate':None},indent=2))
finally:
 if proc:
  proc.stdin.close()
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
 if env:env.close()
 (out/'exit-status.txt').write_text(str(status));app.close()
raise SystemExit(status)
