"""Fixed balanced held-out flow-matching loss; no parameter updates or augmentation."""
import argparse,json,time,hashlib
from pathlib import Path

def balanced_indices(lengths,per_episode=64):
 indices=[];groups=[];offset=0
 for ep,length in enumerate(lengths):
  if length<per_episode:raise ValueError('Episode shorter than requested unique sample count')
  chosen=[offset+int((i+.5)*length/per_episode) for i in range(per_episode)]
  indices.extend(chosen);groups.extend([ep]*per_episode);offset+=length
 return indices,groups

def fixed_noise_time(shape,seed,device):
 import torch
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed)
  noise=torch.randn(shape)
  t=torch.distributions.Beta(torch.tensor(1.5),torch.tensor(1.)).sample((shape[0],))*.999+.001
 return noise.to(device),t.to(device)

def main():
 import torch,numpy as np,jax,gc
 from hydra import initialize_config_dir,compose
 from omegaconf import open_dict
 from openpi.models.model import Observation
 from rlinf_extension import register,build_model
 from rlinf.data.datasets.openpi import build_openpi_sft_dataloader
 p=argparse.ArgumentParser();p.add_argument('--steps',default='3000,6000,9000,12000,15000,18000,20000');p.add_argument('--output',required=True);p.add_argument('--model-override');p.add_argument('--samples-per-episode',type=int,default=64);p.add_argument('--repeats',type=int,default=4);a=p.parse_args()
 root=Path(__file__).resolve().parents[2];run=root/'experiments/phase8-milk-bin-20261108';out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 register();torch.set_num_threads(4);steps=[int(x) for x in a.steps.split(',')]
 episodes=[json.loads(x) for x in (run/'lerobot-validation/meta/episodes.jsonl').read_text().splitlines()]
 assert [e['episode_index'] for e in episodes]==list(range(5))
 ids,groups=balanced_indices([e['length'] for e in episodes],a.samples_per_episode)
 with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent),version_base='1.1'):cfg=compose(config_name='sft')
 def checkpoint(step):return Path(a.model_override) if a.model_override else run/f'sft/milk-bin-pi05-sft/checkpoints/global_step_{step}/actor'
 with open_dict(cfg):
  cfg.data.num_workers=0;cfg.actor.model.model_path=str(checkpoint(steps[0]));cfg.actor.model.precision='bf16'
  cfg.actor.model.openpi.task='eval'  # Match online inference dtypes, without SFT-only FP32 master casts.
  cfg.actor.model.openpi_data.norm_stats_path=str(run/'lerobot-train/norm_stats.json')
 loader,_=build_openpi_sft_dataloader(cfg,1,0,str(run/'lerobot-validation'),eval_dataset=True)
 inner=loader._data_loader._data_loader
 subset=torch.utils.data.Subset(inner.dataset,ids)
 small=torch.utils.data.DataLoader(subset,batch_size=4,shuffle=False,num_workers=0,collate_fn=inner.collate_fn)
 batches=[]
 for batch in small:
  batch=jax.tree.map(torch.as_tensor,batch);batches.append((Observation.from_dict(batch),batch['actions']))
 assert sum(x[1].shape[0] for x in batches)==len(ids)
 manifest={'validation_episodes':5,'samples_per_episode':a.samples_per_episode,'indices':ids,'episode_groups':groups,'noise_repeats':a.repeats,'noise_seed':90200,'augmentation':False,'precision':'native eval BF16 with FP32 islands, same as online policy','train_norm_sha256':hashlib.sha256((run/'lerobot-train/norm_stats.json').read_bytes()).hexdigest(),'purpose':'fixed balanced validation subset, checkpoint selection only'}
 (out/'sample-manifest.json').write_text(json.dumps(manifest,indent=2))
 model=build_model(cfg.actor.model).cuda().eval()
 for param in model.parameters():param.requires_grad_(False)
 results=[];start=time.monotonic()
 for step in steps:
  path=checkpoint(step)/'model_state_dict/full_weights.pt'
  state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
  model.load_state_dict(state,strict=True);del state;gc.collect()
  records=[]
  with torch.inference_mode():
   for repeat in range(a.repeats):
    for b,(obs,act) in enumerate(batches):
     obs=model._observation_to_device(obs);act=model._actions_to_device(act)
     noise,t=fixed_noise_time(act.shape,90200+repeat*10000+b,act.device)
     loss=model.compute_loss(obs,act,train=False,noise=noise,time=t)
     assert loss.shape==act.shape and torch.isfinite(loss).all()
     full=loss.mean(dim=(1,2)).cpu().tolist();executed=loss[:,:5,:7].mean(dim=(1,2)).cpu().tolist()
     for j,(f,e) in enumerate(zip(full,executed)):
      records.append({'repeat':repeat,'sample_index':ids[b*4+j],'episode':groups[b*4+j],'flow_mse':f,'executed_flow_mse':e})
  report={'step':step,'checkpoint':str(checkpoint(step)),'full_flow_mse':float(np.mean([r['flow_mse'] for r in records])),'executed_flow_mse':float(np.mean([r['executed_flow_mse'] for r in records])),'per_episode_flow_mse':{str(ep):float(np.mean([r['flow_mse'] for r in records if r['episode']==ep])) for ep in range(5)},'sample_evaluations':len(records),'frozen_eval_only':True}
  (out/f'step-{step}.json').write_text(json.dumps({'summary':report,'records':records},indent=2));results.append(report)
  (out/'progress.json').write_text(json.dumps(results,indent=2));print(json.dumps(report),flush=True)
 (out/'report.json').write_text(json.dumps({'complete':True,'manifest':manifest,'checkpoints':results,'wall_seconds':time.monotonic()-start},indent=2))

if __name__=='__main__':main()
