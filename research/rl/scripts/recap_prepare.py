"""Prepare independent RECAP metadata views and native RLinf configurations."""
import argparse,copy,json,shutil
from pathlib import Path
import yaml

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['smoke','full'],required=True);p.add_argument('--value-steps',type=int,default=8000);p.add_argument('--cfg-steps',type=int,default=10000);a=p.parse_args()
 root=a.root;source=root/'third_party/RLinf';storage=root/'storage';raw=storage/'datasets/recap-libero-task0'
 out=root/'research/rl/outputs'/('scratch/recap-20261110/smoke' if a.mode=='smoke' else 'runs/libero/recap/pi05-20k-task0-20261110')
 if (out/'provenance/recipe.json').exists():raise FileExistsError('Prepared run already exists; refusing configuration overwrite')
 if a.mode=='full' and not (raw/'verified.json').is_file():raise RuntimeError('Full public dataset must be verified before formal preparation')
 out.mkdir(parents=True,exist_ok=True)
 for name in ['config','logs','metrics','provenance','data','videos']:(out/name).mkdir(exist_ok=True)
 datasets={}
 for name,split in [('sft','sft'),('rollout','train'),('eval','eval')]:
  if a.mode=='smoke' and name!='sft':continue
  src=raw/f'libero10_task0_{split}';dst=out/'data'/name
  if not dst.exists():
   dst.mkdir();shutil.copytree(src/'meta',dst/'meta')
   for folder in ['data','videos']:
    if (src/folder).exists():(dst/folder).symlink_to(src/folder,target_is_directory=True)
  datasets[name]=str(dst)
 paths=[{'dataset_path':datasets['sft'],'type':'sft','weight':1.0}]
 if a.mode=='full':paths.append({'dataset_path':datasets['rollout'],'type':'rollout','weight':1.0})
 tag='fail300_v1';advtag='fail300_N10_q30_v1'
 configs=source/'examples/offline_rl/config'
 def read(name):return yaml.safe_load((configs/(name+'.yaml')).read_text())
 def write(name,c):
  c.setdefault('hydra',{}).setdefault('searchpath',[])
  c['hydra']['searchpath']=[f'file://{configs}']
  (out/'config'/(name+'.yaml')).write_text(yaml.safe_dump(c,sort_keys=False))
 returns=read('recap_compute_returns');returns['data'].update(train_data_paths=paths+([{'dataset_path':datasets['eval'],'type':'rollout'}] if a.mode=='full' else []),tag=tag,num_workers=8);write('returns',returns)
 value=read('recap_value_model_sft');value['runner'].update(max_epochs=-1,max_steps=4 if a.mode=='smoke' else a.value_steps,val_check_interval=2 if a.mode=='smoke' else 1000,save_interval=4 if a.mode=='smoke' else 3000)
 value['runner']['logger'].update(log_path=str(out),experiment_name='value',logger_backends=['tensorboard'])
 value['data'].update(tag=tag,train_num_workers=2,eval_num_workers=2,prefetch_factor=2,train_data_paths=[dict(x,robot_type='libero',model_type='pi05') for x in paths],eval_data_paths=[{'dataset_path':datasets.get('eval',datasets['sft']),'max_samples':64 if a.mode=='smoke' else None,'robot_type':'libero','model_type':'pi05'}])
 value['actor'].update(micro_batch_size=4 if a.mode=='smoke' else 32,global_batch_size=4 if a.mode=='smoke' else 32)
 value['actor']['model'].update(siglip_path=str(storage/'models/siglip2-so400m-patch14-224'),gemma3_path=str(storage/'models/gemma-3-270m'),tokenizer_path=str(storage/'models/gemma-3-270m'))
 value['actor']['optim'].update(total_training_steps=value['runner']['max_steps'],lr_warmup_steps=0 if a.mode=='smoke' else 500)
 write('value',value)
 adv=read('recap_compute_advantages');adv['advantage'].update(value_checkpoint=str(out/'value/checkpoints'/f"global_step_{value['runner']['max_steps']}"/'actor'),batch_size=32,num_dataloader_workers_per_gpu=2,tag=advtag,returns_tag=tag)
 for k in ['siglip_path','gemma3_path','tokenizer_path']:adv['advantage']['model'][k]=value['actor']['model'][k]
 adv['data']['train_data_paths']=[dict(x,robot_type='libero') for x in paths];write('advantages',adv)
 cfg=read('cfg_rl_openpi');cfg['runner'].update(max_epochs=-1,max_steps=2 if a.mode=='smoke' else a.cfg_steps,save_interval=2 if a.mode=='smoke' else 3000)
 cfg['runner']['logger'].update(log_path=str(out),experiment_name='cfg',logger_backends=['tensorboard'])
 cfg['data'].update(num_workers=2,advantage_tag=advtag,train_data_paths=paths)
 cfg['actor'].update(micro_batch_size=4,global_batch_size=4)
 cfg['actor']['model'].update(model_path=str(storage/'models/pi05-fewshot-libero/global_step_20000'),num_action_chunks=5,num_steps=10)
 cfg['actor']['optim'].update(total_training_steps=cfg['runner']['max_steps'],lr_warmup_steps=0 if a.mode=='smoke' else 1000)
 write('cfg',cfg)
 (out/'provenance/recipe.json').write_text(json.dumps({'mode':a.mode,'value_steps':value['runner']['max_steps'],'cfg_steps':cfg['runner']['max_steps'],'policy_batch':4,'value_batch':value['actor']['global_batch_size'],'dataset_note':'SFT contains 3 demonstrations per task for 10 tasks; rollout/eval are Task0 only. Raw metadata is not modified.'},indent=2))
 print(out)
if __name__=='__main__':main()
