"""Select only on saved validation observations, using deployment8-action queue."""
import os,sys,json,time,gc,hashlib
from pathlib import Path
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'adapters/sage_dp'))
sys.path.insert(0,str(ROOT/'upstream/sage/robomimic'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
from bootstrap import disable_unused_language_encoder
disable_unused_language_encoder()
import torch
from robomimic.utils.file_utils import policy_from_checkpoint
from open_loop_metrics import predict_recorded,action_errors,summarize

def main():
 run=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(2);start=time.monotonic()
 data=json.loads((run/'dataset-report.json').read_text());keys=['ee_pose','joint_qpos','rgb_tabletop','rgb_wrist']
 checkpoints=sorted(run.glob('trained/photo_dp_20k/*/models/model_epoch_*.pth'),key=lambda p:int(p.stem.split('_')[-1]))
 assert len(checkpoints)==10
 rows=[]
 with h5py.File(run/'training.hdf5','r') as f:
  valid=[x.decode() for x in f['mask/valid'][:]];assert set(valid)==set(data['valid'])
  assert not set(valid)&set(data['train']+data['test'])
  observations={d:{k:f['data/'+d+'/obs/'+k][:] for k in keys} for d in valid}
  refs={d:f['data/'+d+'/actions'][:] for d in valid}
  for path in checkpoints:
   policy,ckpt=policy_from_checkpoint(ckpt_path=str(path),device=torch.device('cuda:0'))
   shape=ckpt['shape_metadata'];shape=shape[0] if isinstance(shape,list) else shape
   assert set(shape['all_obs_keys'])==set(keys)
   errors=[];arrays={};all_predictions=[];all_references=[]
   for demo in valid:
    for seed in [208,209,210]:
     torch.manual_seed(seed);np.random.seed(seed)
     with torch.inference_mode():pred=predict_recorded(policy,observations[demo],'chunk8')
     assert np.isfinite(pred).all(), 'Nonfinite candidate actions'
     mask=np.linalg.norm(pred[:,3:7],axis=1)>=.01
     if mask.any():errors.append(action_errors(pred[mask],refs[demo][mask]))
     all_predictions.append(pred);all_references.append(refs[demo]);arrays[f'{demo}_seed{seed}']=pred
   m=summarize({k:np.concatenate([e[k] for e in errors]) for k in errors[0]}) if errors else {}
   all_p=np.concatenate(all_predictions);all_r=np.concatenate(all_references)
   delta=(all_p[:,:3]-all_r[:,:3])*100;distance=np.linalg.norm(delta,axis=1)
   m.update(frames=len(all_p),position_mean_cm=float(distance.mean()),position_rmse_cm=float(np.sqrt(np.mean(distance**2))),position_p95_cm=float(np.percentile(distance,95)),xyz_rmse_cm=np.sqrt(np.mean(delta**2,axis=0)).tolist(),gripper_accuracy=float(np.mean((all_p[:,7]>=0)==(all_r[:,7]>=0))),invalid_quaternion_frames=int(np.sum(np.linalg.norm(all_p[:,3:7],axis=1)<.01)),rotation_metric_scope='rotation and secondary per-class gripper metrics use quaternion norms>=0.01; position metrics and overall gripper_accuracy use all predictions')
   row={'checkpoint':str(path),'epoch':int(path.stem.split('_')[-1]),'optimizer_steps':int(path.stem.split('_')[-1])*50,'metrics':m}
   rows.append(row);np.savez_compressed(out/(path.stem+'-validation.npz'),**arrays)
   print(json.dumps(row),flush=True)
   del policy,ckpt;gc.collect();torch.cuda.empty_cache()
 selected=min(rows,key=lambda x:(x['metrics']['position_mean_cm'],x['epoch']))
 h=hashlib.sha256()
 with Path(selected['checkpoint']).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 report={'complete':True,'selection_rule':'minimum validation mean position error; earlier epoch breaks ties',
  'validation_episodes':valid,'seeds':[208,209,210],'mode':'original8-action queue, recorded observations only',
  'results':rows,'selected':selected,'selected_sha256':h.hexdigest(),'wall_seconds':time.monotonic()-start}
 (out/'selection.json').write_text(json.dumps(report,indent=2)+'\n');(out/'selected-checkpoint.txt').write_text(selected['checkpoint']+'\n')
 (out/'select_checkpoint-executed.py').write_bytes(Path(__file__).read_bytes())
if __name__=='__main__':main()
