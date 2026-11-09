"""Camera-only diagnostic: fixed arm, open/closed fingers; no policy evaluation."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
from sim_task import create_task
import torch,numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from sim_task import ROOT
model,app=create_task(20001,str(out))
try:
 model.reset(seed=20001);env=model.env;robot=env.scene['robot'];offset=env.cfg.scene.camera_wrist.offset
 original_pos=list(offset.pos);original_rot=list(offset.rot)
 from omni.isaac.lab_tasks.manager_based.manipulation.lift_single_obj_scene.config.franka.joint_pos_env_cfg_teaser import build_camera
 profiles=[('original',original_pos,original_rot)]
 for name,pos,target in [('side55',[.055,0,.035],[0,0,.18]),('side70',[.070,0,.035],[0,0,.20])]:
  pos,quat=build_camera(pos,target,[0,1,0]);profiles.append((name,list(pos),list(quat)))
 trace=json.loads((ROOT/'experiments/phase8-milk-bin-20261108/baseline-calibration/episode-001-trace.json').read_text())
 # Reconstruct the first empty-grasp neighborhood from saved actions, without loading a policy.
 for item in trace[:170]:model.step(torch.tensor([item['action']],device=env.device))
 hold=model._pose().clone();q_fixed=robot.data.joint_pos.clone();q_fixed[:,-2:]=.04
 rows=[]
 for name,pos,quat in profiles:
  offset.pos=tuple(pos);offset.rot=tuple(quat)
  for state,grip in [('open',1.),('closed',-1.)]:
   robot.write_joint_state_to_sim(q_fixed,torch.zeros_like(q_fixed));robot.reset();env.action_manager.reset()
   action=torch.cat([hold,torch.tensor([[grip]],device=env.device)],dim=-1)
   for _ in range(60):env.step(action)
   for _ in range(8):env.sim.render()
   for camera in ['wrist','tabletop']:
    sensor=env.scene['camera_'+camera];sensor.update(0.,force_recompute=True)
    Image.fromarray(sensor.data.output['rgb'][0].cpu().numpy()).save(out/f'{name}-{state}-{camera}.png')
   wrist=env.scene['camera_wrist'];rgb=wrist.data.output['rgb'][0].cpu().numpy()
   rows.append({'profile':name,'gripper':state,'offset_pos':pos,'offset_rot_wxyz':quat,'finger_q':robot.data.joint_pos[0,-2:].cpu().tolist(),'hand_base_pose':model._pose()[0].cpu().tolist(),'camera_world_pos':wrist.data.pos_w[0].cpu().tolist(),'camera_ros_quaternion':wrist.data.quat_w_ros[0].cpu().tolist(),'pixel_std':float(rgb.std())})
 (out/'report.json').write_text(json.dumps({'diagnostic_only':True,'policy_loaded':False,'saved_action_replay_steps':170,'seed':20001,'original_offset':original_pos,'comparisons':rows},indent=2))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(3,2,figsize=(8,12),layout='constrained')
 for i,(name,pos,quat) in enumerate(profiles):
  for j,state in enumerate(['open','closed']):
   axs[i,j].imshow(Image.open(out/f'{name}-{state}-wrist.png'));axs[i,j].set_title(f'{name}: {state}');axs[i,j].axis('off')
 fig.savefig(out/'comparison.png',dpi=160)
finally:model.close();app.close()
