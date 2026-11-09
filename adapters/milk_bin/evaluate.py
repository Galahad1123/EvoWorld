"""Seeded online evaluation using the same native RLinf environment/action adapter."""
import argparse,json,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
 import torch,numpy as np,imageio.v2 as imageio
 from hydra import initialize_config_dir,compose
 from omegaconf import OmegaConf,open_dict
 from rlinf_extension import register,TASK,build_model
 register()
 from rlinf.envs import get_env_cls
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed-start',type=int,default=20000);p.add_argument('--horizon',type=int,default=650);p.add_argument('--video-count',type=int,default=3);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
 with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent),version_base='1.1'):sft=compose(config_name='sft')
 model_cfg=sft.actor.model
 with open_dict(model_cfg):
  model_cfg.model_path=str(Path(a.checkpoint).resolve());model_cfg.precision='bf16';model_cfg.openpi.task='eval'
  model_cfg.num_action_chunks=5;model_cfg.openpi.action_chunk=5
  model_cfg.openpi_data.norm_stats_path=str(Path(a.dataset).resolve()/'norm_stats.json')
 from evoworld.config import load
 cfg=OmegaConf.load(Path(load()['upstreams']['RLinf'])/'examples/embodiment/config/env/isaaclab_stack_cube.yaml')
 with open_dict(cfg):
  cfg.init_params.id='EvoWorld-Milk-Bin-Franka-v0';cfg.init_params.task_description=TASK;cfg.init_params.output_dir=str(out/'sim')
  cfg.total_num_envs=cfg.init_params.num_envs=1;cfg.max_episode_steps=a.horizon;cfg.auto_reset=False;cfg.ignore_terminations=False
  cfg.video_cfg.save_video=False;cfg.video_cfg.video_base_dir=str(out)
 env=None;results=[];started=time.monotonic()
 try:
  model=build_model(model_cfg).cuda().eval()
  for param in model.parameters():param.requires_grad_(False)
  env=get_env_cls('isaaclab',cfg)(cfg,1,0,1,None)
  for ep in range(a.episodes):
   seed=a.seed_start+ep;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
   obs,_=env.reset(seed=seed);success=False;steps=0;latencies=[];records=[]
   writer=imageio.get_writer(str(out/f'episode-{ep:03}.mp4'),fps=50) if ep<a.video_count else None
   try:
    while steps<a.horizon:
     begin=time.monotonic()
     with torch.inference_mode():actions,_=model.predict_action_batch(obs,mode='eval')
     torch.cuda.synchronize();latencies.append(time.monotonic()-begin)
     assert actions.shape==(1,5,7) and torch.isfinite(actions).all()
     done=False
     for j in range(min(5,a.horizon-steps)):
      if writer:writer.append_data(np.concatenate([obs['main_images'][0].cpu().numpy(),obs['wrist_images'][0].cpu().numpy()],axis=1))
      action=actions[:,j];obs,reward,term,trunc,info=env.step(action,auto_reset=False);steps+=1
      records.append({'step':steps,'action':action[0].cpu().tolist(),'reward':float(reward[0]),'terminated':bool(term[0]),'truncated':bool(trunc[0])})
      if term.any() or trunc.any():
       success=bool(term[0]);done=True;break
     if done:break
   finally:
    if writer:writer.close()
   result={'episode':ep,'seed':seed,'success':success,'steps':steps,'mean_chunk_latency_s':float(np.mean(latencies)),'chunks':len(latencies)}
   (out/f'episode-{ep:03}-trace.json').write_text(json.dumps(records))
   results.append(result);(out/'progress.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
 finally:
  if env is not None:gc.collect();torch.cuda.synchronize();env.close()
 successes=sum(r['success'] for r in results);n=len(results);z=1.959963984540054
 fraction=successes/n;center=(fraction+z*z/(2*n))/(1+z*z/n);radius=z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))/(1+z*z/n)
 report={'complete':True,'checkpoint':str(Path(a.checkpoint).resolve()),'episodes':results,'successes':successes,'total':n,'success_rate':fraction,'wilson95':[center-radius,center+radius],'action_horizon':20,'executed_chunk':5,'denoise_steps':10,'max_control_steps':a.horizon,'wall_seconds':time.monotonic()-started,'benchmark_valid':a.horizon==650,'peak_policy_process_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='episodes'}),flush=True)
