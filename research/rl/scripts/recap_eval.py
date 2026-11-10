"""Independent Task0 evaluation using the native CFG policy and LiberoEnv."""
import argparse,json,os,time
from pathlib import Path
import numpy as np

def model_config(src,checkpoint,guidance):
 from omegaconf import OmegaConf
 m=OmegaConf.load(src/'examples/offline_rl/config/model/pi0_5.yaml')
 m.model_path=str(checkpoint);m.num_action_chunks=5;m.num_steps=10
 m.openpi=OmegaConf.merge(m.openpi,{'train_expert_only':False,'guidance_type':guidance,'positive_only_conditional':True,'unconditional_prob':.1,'cfgrl_guidance_scale':1.0})
 holder=OmegaConf.create({'actor':{'model':m}});OmegaConf.resolve(holder);m=holder.actor.model
 return m

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--guidance',choices=['no_guide','positive'],required=True);p.add_argument('--episodes',type=int,default=50);p.add_argument('--num-envs',type=int,default=10);a=p.parse_args()
 if a.episodes%a.num_envs or not 1<=a.episodes<=50:raise ValueError('episodes must divide into batches and be <= 50')
 out=a.output
 if out.exists():raise FileExistsError('Evaluation output must be new')
 out.mkdir(parents=True)
 for d in ['config','metrics','videos']:(out/d).mkdir()
 import torch,imageio.v2 as imageio
 from omegaconf import OmegaConf
 from rlinf.envs.sim.libero.libero_env import LiberoEnv
 from rlinf.models.embodiment.openpi_cfg import get_model
 src=a.root/'third_party/RLinf'
 c=OmegaConf.load(src/'examples/embodiment/config/env/libero_10.yaml')
 c=OmegaConf.merge(c,{'task_id_filter':[0],'total_num_envs':a.num_envs,'auto_reset':True,'ignore_terminations':True,'is_eval':True,'use_fixed_reset_state_ids':True,'use_ordered_reset_state_ids':True,'max_episode_steps':520,'max_steps_per_rollout_epoch':520*(a.episodes//a.num_envs),'share_memory':False,'skip_intermediate_renders':True,'video_cfg':{'save_video':False,'video_base_dir':str(out/'videos')}})
 m=model_config(src,a.checkpoint,a.guidance)
 OmegaConf.save(OmegaConf.create({'env':c,'model':m}),out/'config/resolved.yaml')
 torch.set_num_threads(4);env=LiberoEnv(c,a.num_envs,0,1,None);writers={};records=[];start=time.monotonic()
 try:
  model=get_model(m).cuda().eval();torch.manual_seed(195);torch.cuda.manual_seed_all(195)
  obs,_=env.reset()
  with (out/'metrics/episodes.jsonl').open('w') as f:
   for step in range(104*(a.episodes//a.num_envs)):
    tasks=list(env.task_ids);trials=list(env.trial_ids)
    if step<104:
     for i in range(min(a.num_envs,4)):
      if i not in writers:writers[i]=imageio.get_writer(out/'videos'/f'trial-{trials[i]:02d}.mp4',fps=4)
      writers[i].append_data(np.concatenate([obs['main_images'][i].numpy(),obs['wrist_images'][i].numpy()],axis=1))
    elif writers:
     for w in writers.values():w.close()
     writers.clear()
    with torch.inference_mode():actions,_=model.predict_action_batch(obs,mode='eval')
    actions=np.asarray(actions);assert actions.shape==(a.num_envs,5,7) and np.isfinite(actions).all()
    actions=actions.copy();actions[...,-1]=np.where(actions[...,-1]>=.5,-1.,1.)
    observations,_,_,_,infos=env.chunk_step(actions);obs=observations[-1];info=infos[-1]
    if 'final_info' in info:
     metrics=info['final_info']['episode']
     for i in np.flatnonzero(np.asarray(info['_final_info'])):
      row={'task_id':int(tasks[i]),'trial_id':int(trials[i]),'success':bool(metrics['success_once'][i]),'steps':int(metrics['episode_len'][i])};records.append(row);f.write(json.dumps(row)+'\n')
     f.flush();print('episodes',len(records),'successes',sum(x['success'] for x in records),flush=True)
  assert len(records)==a.episodes and {x['trial_id'] for x in records}==set(range(a.episodes)) and all(x['task_id']==0 for x in records)
  summary={'episodes':len(records),'successes':sum(x['success'] for x in records),'success_rate':sum(x['success'] for x in records)/len(records),'guidance':a.guidance,'checkpoint':str(a.checkpoint),'seconds':time.monotonic()-start,'max_gpu_allocated':torch.cuda.max_memory_allocated()}
  (out/'metrics/summary.json').write_text(json.dumps(summary,indent=2));print(summary)
 finally:
  for w in writers.values():w.close()
  env.env.close()
if __name__=='__main__':main()
