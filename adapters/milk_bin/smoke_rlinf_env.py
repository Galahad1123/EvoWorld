"""Verify RLinf's real subprocess transport, observation wrapper and chunk API."""
from pathlib import Path
import argparse,json,gc
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
 import torch
 from omegaconf import OmegaConf,open_dict
 from rlinf_extension import register,TASK
 register()
 from rlinf.envs import get_env_cls
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
 cfg=OmegaConf.load(ROOT/'upstream/RLinf/examples/embodiment/config/env/isaaclab_stack_cube.yaml')
 with open_dict(cfg):
  cfg.init_params.id='EvoWorld-Milk-Bin-Franka-v0';cfg.init_params.task_description=TASK;cfg.init_params.output_dir=str(out)
  cfg.total_num_envs=cfg.init_params.num_envs=1;cfg.max_episode_steps=650;cfg.auto_reset=True;cfg.ignore_terminations=False
  cfg.video_cfg.save_video=False;cfg.video_cfg.video_base_dir=str(out)
 env=get_env_cls('isaaclab',cfg)(cfg,1,0,1,None)
 try:
  obs,_=env.reset(seed=810)
  actions=torch.zeros((1,5,7),device=env.device);actions[:,:,-1]=1
  obs_list,rewards,term,trunc,infos=env.chunk_step(actions)
  assert rewards.shape==(1,5) and torch.allclose(rewards,torch.full_like(rewards,-.001))
  assert not term.any() and not trunc.any()
  for key in ['main_images','wrist_images']:assert obs_list[-1][key].shape==(1,224,224,3)
  assert obs_list[-1]['states'].shape==(1,8)
  env.reset(seed=811)
  report={'passed':True,'native_rlinf_wrapper':True,'chunk_steps':5,'reset':True,'observations':{k:list(v.shape) for k,v in obs.items() if torch.is_tensor(v)}}
 finally:
  gc.collect();torch.cuda.synchronize();env.close()
 report['close_completed']=True;(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
