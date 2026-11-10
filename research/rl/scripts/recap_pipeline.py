"""Fail-closed, sequential RECAP baseline controller. No stage runs after a failure."""
import argparse,json,os,shutil,subprocess,time,fcntl
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--value-steps',type=int,default=8000);p.add_argument('--cfg-steps',type=int,default=10000);p.add_argument('--resume',action='store_true');p.add_argument('--download-pid',type=int);a=p.parse_args()
 root=a.root.resolve();scripts=Path(__file__).resolve().parent;py=root/'storage/envs/rlinf-libero-py312/bin/python';out=root/'research/rl/outputs/runs/libero/recap/pi05-20k-task0-20261110'
 out.mkdir(parents=True,exist_ok=True)
 status=out/'pipeline-status.json'
 lock=(out/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if status.exists() and not a.resume:raise FileExistsError('Existing pipeline status; use --resume')
 if (out/'metrics/summary.json').exists():
  print('Pipeline already complete');return
 if (out/'provenance/recipe.json').exists():
  recipe=json.loads((out/'provenance/recipe.json').read_text())
  if recipe['value_steps']!=a.value_steps or recipe['cfg_steps']!=a.cfg_steps:raise ValueError('Resume budget differs from immutable recipe')
 def record(stage,**kwargs):status.write_text(json.dumps({'stage':stage,'time':time.time(),**kwargs},indent=2));print(stage,kwargs,flush=True)
 def execute(cmd,stage,env=None):
  record(stage,command=list(map(str,cmd)))
  (out/'logs').mkdir(exist_ok=True)
  with (out/'logs'/f'controller-{stage}.log').open('w') as log:
   proc=subprocess.Popen(list(map(str,cmd)),env=env,stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    with (out/'resources.jsonl').open('a') as f:
     gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
     f.write(json.dumps({'time':time.time(),'stage':stage,'gpu':gpu.stdout.strip(),'disk_free':shutil.disk_usage(root).free})+'\n')
    time.sleep(15)
   if proc.returncode:raise RuntimeError(f'{stage} exited {proc.returncode}; see {log.name}')
 try:
  record('waiting_for_verified_resources')
  receipts=[root/'storage/datasets/recap-libero-task0/verified.json',root/'storage/models/gemma-3-270m/verified.json',root/'storage/models/siglip2-so400m-patch14-224/verified.json']
  deadline=time.time()+12*3600
  while not all(x.is_file() for x in receipts):
   if a.download_pid:
    try:os.kill(a.download_pid,0)
    except ProcessLookupError:raise RuntimeError('Downloader exited without verified resources')
   if time.time()>deadline:raise TimeoutError('Resource download not verified within 12h')
   time.sleep(30)
  smoke=root/'research/rl/outputs/scratch/recap-20261110/smoke'
  assert json.loads((smoke/'metrics/acceptance.json').read_text())['passed'], 'Smoke acceptance required'
  if not (out/'provenance/recipe.json').exists():execute([py,scripts/'recap_prepare.py','--root',root,'--mode','full','--value-steps',a.value_steps,'--cfg-steps',a.cfg_steps],'prepare')
  for stage in ['returns','value','advantages','cfg']:execute([py,scripts/'recap_stage.py','--root',root,'--run',out,'--stage',stage,*(['--resume'] if a.resume else [])],stage)
  package=out/'policy';package.mkdir(exist_ok=True)
  trained=out/f'cfg/checkpoints/global_step_{a.cfg_steps}/actor/model_state_dict'
  assert (trained/'full_weights.pt').is_file()
  if not (package/'model_state_dict').exists():(package/'model_state_dict').symlink_to(trained,target_is_directory=True)
  base=root/'storage/models/pi05-fewshot-libero/global_step_20000'
  shutil.copytree(base/'physical-intelligence',package/'physical-intelligence',dirs_exist_ok=True)
  env=dict(os.environ);env.update(PYTHONPATH=str(root/'third_party/RLinf'),HF_HOME=str(root/'storage/cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBERO_TYPE='standard',ROBOT_PLATFORM='LIBERO',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
  for label,checkpoint,guidance in [('baseline',base,'no_guide'),('final',package,'positive')]:
   evaluation=out/f'evaluation-{label}'
   if (evaluation/'metrics/summary.json').is_file():continue
   if evaluation.exists():evaluation.rename(out/'provenance'/f'failed-evaluation-{label}-{time.time_ns()}')
   execute([py,scripts/'recap_eval.py','--root',root,'--checkpoint',checkpoint,'--output',out/f'evaluation-{label}','--guidance',guidance],f'evaluation-{label}',env)
  b=json.loads((out/'evaluation-baseline/metrics/summary.json').read_text());f=json.loads((out/'evaluation-final/metrics/summary.json').read_text())
  summary={'complete':True,'baseline':b,'recap':f,'delta_percentage_points':100*(f['success_rate']-b['success_rate']),'value_steps':a.value_steps,'cfg_steps':a.cfg_steps,'notes':['Public SFT is 3 demonstrations per task across 10 tasks; rollout/eval dataset is Task0.','This uses 20k initialization; not a numerical reproduction of published 48.8% baseline.','Independent simulator evaluation uses 50 fixed Task0 reset states, 520 steps, action chunk 5.']}
  (out/'metrics/summary.json').write_text(json.dumps(summary,indent=2))
  (out/'README.md').write_text(f"# RECAP Task 0 基线\n\n初始化：{b['successes']}/50；RECAP：{f['successes']}/50。\n\n完整配置见 config，指标见 metrics/summary.json，视频在 evaluation-baseline/videos 与 evaluation-final/videos。\n")
  record('complete',summary='metrics/summary.json')
 except Exception as exc:
  record('failed',error=str(exc));raise
if __name__=='__main__':main()
