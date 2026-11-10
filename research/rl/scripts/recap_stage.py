"""Run one native RECAP stage with explicit environment, provenance and logs."""
import argparse,json,os,subprocess,time
from pathlib import Path
ENTRIES={'returns':'examples/offline_rl/advantage_labeling/recap/process/compute_returns.py','value':'examples/offline_rl/advantage_labeling/recap/train_value.py','advantages':'examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py','cfg':'examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=ENTRIES,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 root=a.root.resolve();out=a.run.resolve();src=root/'third_party/RLinf';py=root/'storage/envs/rlinf-libero-py312/bin/python'
 assert out.is_relative_to(root/'research/rl/outputs')
 for name in ['logs','provenance','metrics']:(out/name).mkdir(parents=True,exist_ok=True)
 import fcntl
 lock=(out/'provenance'/f'{a.stage}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 record=out/'provenance'/f'{a.stage}-execution.json'
 if record.exists() and json.loads(record.read_text()).get('exit_code')==0:
  if a.resume:
   print('Stage already verified complete');return
  raise SystemExit('Stage already succeeded; refusing overwrite')
 env=dict(os.environ);env.update(PYTHONPATH=str(src),REPO_PATH=str(src),EMBODIED_PATH=str(src/'examples/embodiment'),HF_ENDPOINT='https://hf-mirror.com',HF_HOME=str(root/'storage/cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled')
 libs=root/'storage/runtime/ffmpeg/usr/lib/x86_64-linux-gnu'
 env['LD_LIBRARY_PATH']=os.pathsep.join([str(libs),str(libs/'pulseaudio'),str(root/'storage/runtime/usr/lib/x86_64-linux-gnu'),env.get('LD_LIBRARY_PATH','')])
 cmd=[str(py),str(src/ENTRIES[a.stage]),'--config-path',str(out/'config'),'--config-name',a.stage]
 if a.resume and a.stage in ('value','cfg'):
  checkpoints=[x for x in (out/a.stage/'checkpoints').glob('global_step_*') if (x/'actor/dcp_checkpoint/.metadata').is_file() and (x/'actor/model_state_dict/full_weights.pt').is_file()]
  if checkpoints:
   latest=max(checkpoints,key=lambda x:int(x.name.split('_')[-1]));cmd.append(f'++runner.resume_dir={latest}')
 if a.stage=='advantages':cmd=[str(py),'-m','torch.distributed.run','--standalone','--nproc_per_node=1',*cmd[1:]]
 info={'command':cmd,'started':time.time(),'pid':os.getpid(),'evoworld_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'rlinf_commit':subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()};record.write_text(json.dumps(info,indent=2))
 with (out/'logs'/f'{a.stage}.log').open('a') as f:
  code=subprocess.call(cmd,env=env,cwd=out/'logs',stdout=f,stderr=subprocess.STDOUT)
 info.update(exit_code=code,finished=time.time());record.write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
 raise SystemExit(code)
if __name__=='__main__':main()
