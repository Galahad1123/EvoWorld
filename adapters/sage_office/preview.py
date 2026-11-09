"""Render an office background with the unchanged native photo task and Franka."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--background',required=True);p.add_argument('--output',required=True);p.add_argument('--full-room',action='store_true');p.add_argument('--tabletop-profile')
from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import hashlib,json,os,sys,traceback
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'adapters/sage_photo_dp'))
import photo_runtime as photo
import runtime as office
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.manager_based.manipulation.lift_single_obj_scene.config.franka.ik_abs_env_cfg import FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False);env=None;status=1
try:
 run=ROOT/'experiments/phase8-photo-train-EEtcZBBs';data=json.loads((run/'dataset-report.json').read_text());meta=data['episodes'][data['test'][0]]
 initial=json.loads((Path(meta['source_run'])/f"initial-{meta['attempt']:03}.json").read_text())
 cfg=FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser(config_yaml=str(next(run.glob('robomimic_data/*/params/env_init.yaml'))))
 photo.configure(cfg,408);settings=office.configure(cfg,args.background,args.tabletop_profile)
 if args.full_room:
  settings['overview_position']=[-3.0,-3.3,7.6];settings['overview_target']=[2.08,1.97,.8]
  settings['display_mode']='full-room cutaway; south/west walls hidden for first image, restored for topdown'
  cfg.scene.camera_office_overview.width=1600;cfg.scene.camera_office_overview.height=1200
  cfg.scene.camera_office_overview.spawn.focal_length=40.;cfg.scene.camera_office_overview.spawn.horizontal_aperture=36.

 cfg.scene.num_envs=1;cfg.sim.device='cuda:0';cfg.terminations.time_out=None
 for key in list(vars(cfg.scene)):
  if key.startswith('camera_') and key not in {'camera_left','camera_right','camera_wrist','camera_tabletop','camera_tabletop_preview','camera_office_overview'}:setattr(cfg.scene,key,None)
 for key in list(vars(cfg.observations.policy)):
  if '_vis' in key:setattr(cfg.observations.policy,key,None)
 os.chdir(out);env=ManagerBasedRLEnv(cfg=cfg);photo.attach(env);office.attach(env,settings)
 profile_path=out/'scene-profile.json';profile=json.loads(profile_path.read_text());profile['previous_room']=profile['room'];profile['room']=settings['background_manifest']['layout_id'];profile['background']=settings;profile_path.write_text(json.dumps(profile,indent=2))
 tensor=lambda x:torch.as_tensor(x,device=env.device,dtype=torch.float32)
 robot=env.scene['robot'];env.reset(seed=408)
 robot.write_root_link_state_to_sim(tensor(initial['robot_root']));robot.write_joint_state_to_sim(tensor(initial['q']),tensor(initial['qd']));robot.reset()
 for name,values in initial['objects'].items():
  if name in env.scene.rigid_objects:
   env.scene[name].write_root_state_to_sim(tensor(values));env.scene[name].reset()
 env.action_manager.reset();env.action_manager.process_action(tensor(initial['previous_action']))
 for name in ['left','right','tabletop']:
  c=initial['cameras'][name];env.scene['camera_'+name].set_world_poses(tensor(c['position']),tensor(c['quaternion']),convention='ros')
 office.attach(env,settings)
 env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
 ee=env.scene['ee_frame'].data;hold=torch.cat([ee.target_pos_source.reshape(1,3),ee.target_quat_source.reshape(1,4),torch.ones((1,1),device=env.device)],dim=1)
 states=[]
 for _ in range(100):
  env.step(hold);states.append(photo.old.cpu(env.scene['milk'].data.root_state_w[0]).copy())
 from pxr import UsdGeom
 import omni.usd
 stage=omni.usd.get_context().get_stage();hidden=[]
 if args.full_room:
  for prim in stage.Traverse():
   name=prim.GetName()
   if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).startswith('/World/envs/env_0/OfficeBackground/') and name.startswith('wall_') and ('_south_' in name or '_west_' in name):
    UsdGeom.Imageable(prim).MakeInvisible();hidden.append(str(prim.GetPath()))
  assert len(hidden)==2,hidden
  settings['display_only_hidden_walls']=hidden
 for _ in range(12):env.sim.render()
 images={}
 camera_states={}
 for name in ['office_overview','tabletop','wrist']+(['tabletop_preview'] if args.tabletop_profile else []):
  camera=env.scene['camera_'+name];camera.update(0.,force_recompute=True);pixels=photo.old.cpu(camera.data.output['rgb'][0])
  camera_states[name]={'position':photo.old.cpu(camera.data.pos_w).tolist(),'quaternion_ros':photo.old.cpu(camera.data.quat_w_ros).tolist(),'intrinsics':photo.old.cpu(camera.data.intrinsic_matrices).tolist()}
  assert pixels.std()>5;Image.fromarray(pixels).save(out/(name+'.png'));images[name]={'shape':list(pixels.shape),'std':float(pixels.std())}
 if args.full_room:
  for path in hidden:UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeVisible()
  camera=env.scene['camera_office_overview']
  camera.set_world_poses_from_view(tensor([[2.08,1.969,10.5]]),tensor([[2.08,1.97,0.0]]))
  for _ in range(12):env.sim.render()
  camera.update(0.,force_recompute=True);pixels=photo.old.cpu(camera.data.output['rgb'][0])
  Image.fromarray(pixels).save(out/'room_topdown.png');images['room_topdown']={'shape':list(pixels.shape),'std':float(pixels.std())}
  settings['topdown_position']=[2.08,1.969,10.5];settings['topdown_target']=[2.08,1.97,0.0]
  settings['walls_restored_after_cutaway']=True
 np.savez_compressed(out/'hold-state.npz',milk=np.array(states),q=photo.old.cpu(robot.data.joint_pos[0]))
 displacement=float(np.linalg.norm(np.array(states)[:,:3]-np.array(states)[0,:3],axis=1).max());assert displacement<.005
 from pxr import Usd,UsdGeom,UsdPhysics
 import omni.usd
 stage=omni.usd.get_context().get_stage();stage.GetRootLayer().Export(str(out/'composed-stage.usda'))
 assert stage.GetPrimAtPath('/World/envs/env_0/OfficeBackground').IsValid()
 asset_hashes={str(f.relative_to(photo.BUNDLE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in photo.BUNDLE.rglob('*') if f.is_file()}
 report={'complete':True,'purpose':'office background preview and100-step hold smoke; no policy evaluation/retraining','settings':settings,'images':images,'hold_steps':100,'hold_seconds':100*env.step_dt,'milk_max_displacement_m':displacement,'native_bundle_hashes':asset_hashes,'initial_fixture':data['test'][0],'camera_states':camera_states,'policy_cameras':('tabletop camera uses explicit profile; policy resolution and wrist unchanged; see camera_states for actual intrinsics/extrinsics' if args.tabletop_profile else 'unchanged resolution/intrinsics/extrinsics; appearance changed'),'previous_policy_success_rate_applies':False}
 (out/'report.json').write_text(json.dumps(report,indent=2));status=0
 for name in ['preview.py','runtime.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
 print(json.dumps({'complete':True,'milk_max_displacement_m':displacement,'output':str(out)}),flush=True)
except BaseException:
 (out/'error.txt').write_text(traceback.format_exc());raise
finally:
 if env:env.close()
 (out/'exit-status.txt').write_text(str(status));app.close()
