"""Checked SAGE demonstrations to the LeRobot version installed with RLinf.

PNG images avoid lossy video/decoder dependencies. Original HDF5 stays immutable.
Only a training split can write normalization statistics.
"""
import argparse,hashlib,json
from pathlib import Path
import h5py
import numpy as np
from actions import absolute_to_delta,delta_to_absolute
from rlinf_extension import TASK

def file_hash(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def export(run,out,count,split):
 from lerobot.datasets.lerobot_dataset import LeRobotDataset
 run=Path(run).resolve();out=Path(out).resolve()
 if out.exists():raise FileExistsError(out)
 verification=json.loads((run/'collection-verification.json').read_text());assert verification['passed'] and verification['count']==count
 source=next(run.glob('robomimic_data/*/experts.hdf5'))
 mapping=[json.loads(x) for x in (run/'demo-attempts.jsonl').read_text().splitlines()]
 accepted={r['attempt']:r for r in map(json.loads,(run/'physical-acceptance.jsonl').read_text().splitlines())}
 pre={(r['attempt'],r['iteration']):r for r in map(json.loads,(run/'pre-action-state.jsonl').read_text().splitlines())}
 dt=json.loads((run/'scene-profile.json').read_text())['step_dt'];fps=round(1/dt)
 assert abs(fps*dt-1)<1e-6 and len(mapping)==count
 features={
  'observation.images.front':{'dtype':'image','shape':(224,224,3),'names':['height','width','channels']},
  'observation.images.wrist':{'dtype':'image','shape':(224,224,3),'names':['height','width','channels']},
  'observation.state':{'dtype':'float32','shape':(8,),'names':[f'joint{i}' for i in range(1,8)]+['finger_width_m']},
  'action':{'dtype':'float32','shape':(7,),'names':['dx_m','dy_m','dz_m','drx_rad','dry_rad','drz_rad','gripper_open_positive']}}
 dataset=LeRobotDataset.create(repo_id='evoworld/milk_bin_'+split,root=out,fps=fps,robot_type='franka',features=features,use_videos=False,image_writer_threads=4)
 stats={'state':[],'actions':[]};episodes=[]
 with h5py.File(source) as f:
  assert len(f['data'])==count
  for item in mapping:
   demo=f"demo_{item['demo']}";g=f['data'][demo];a=item['attempt'];assert accepted[a]['success']
   abs_actions=g['actions'][:];n=len(abs_actions)
   rows=[pre[(a,t)] for t in range(21,21+n)]
   pose=np.asarray([r['ee_base'] for r in rows]);q=np.asarray([r['joint_pos'] for r in rows])
   np.testing.assert_allclose(q[:,:7],g['obs/joint_qpos'][:,:7],atol=1e-6)
   np.testing.assert_allclose(pose,g['obs/ee_pose'][:,:7],atol=1e-6)
   delta=absolute_to_delta(pose,abs_actions)
   recon=delta_to_absolute(pose,delta)
   np.testing.assert_allclose(recon[:,:3],abs_actions[:,:3],atol=1e-6)
   quat_dot=np.abs(np.sum(recon[:,3:7]*abs_actions[:,3:7],axis=-1));assert np.allclose(quat_dot,1,atol=1e-5)
   state=np.c_[q[:,:7],q[:,7:9].sum(axis=-1)].astype(np.float32)
   stats['state'].append(state);stats['actions'].append(delta)
   for t in range(n):
    dataset.add_frame({'observation.images.front':g['obs/rgb_tabletop'][t],
     'observation.images.wrist':g['obs/rgb_wrist'][t],'observation.state':state[t],'action':delta[t]},task=TASK)
   dataset.save_episode();episodes.append({'demo':demo,'attempt':a,'frames':n,'acceptance':accepted[a]})
   print('EXPORTED',demo,n,flush=True)
 dataset.stop_image_writer()
 if split=='train':
  normalization={}
  for key,values in stats.items():
   x=np.concatenate(values);normalization[key]={'mean':x.mean(0).tolist(),'std':x.std(0).tolist(),'q01':np.quantile(x,.01,axis=0).tolist(),'q99':np.quantile(x,.99,axis=0).tolist()}
  (out/'norm_stats.json').write_text(json.dumps({'norm_stats':normalization},indent=2))
 # Reload through installed official reader, including action-window construction.
 loaded=LeRobotDataset(repo_id='evoworld/milk_bin_'+split,root=out,delta_timestamps={'action':[t/fps for t in range(20)]})
 offset=0
 for record,actions in zip(episodes,stats['actions']):
  row=loaded[offset]
  np.testing.assert_allclose(row['action'][0].numpy(),actions[0],atol=1e-7)
  assert tuple(row['observation.images.front'].shape)==(3,224,224)
  offset+=record['frames']
 assert len(loaded)==offset
 report={'complete':True,'source_hdf5':str(source),'source_sha256':file_hash(source),'split':split,'episodes':episodes,'frames':offset,'fps':fps,'action_contract':'robot-base metres + spatial rotation vector radians; +1 open/-1 close','state_contract':'7 measured arm joints, sum of measured finger positions','camera_encoding':'lossless PNG','reader_verified':True,'normalization':'this training split only' if split=='train' else 'none; use training stats'}
 (out/'source-manifest.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--count',required=True,type=int);p.add_argument('--split',choices=['train','validation'],required=True);a=p.parse_args();export(a.run,a.output,a.count,a.split)
