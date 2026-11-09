"""Read-only SFT gate and persistent, once-only cross-host DSRL handoff."""
import argparse,base64,fcntl,json,os,shlex,subprocess,sys,time
from pathlib import Path
ROOT='/home/hello/lizk/vlas/EvoWorld'
RUN=ROOT+'/experiments/phase8-milk-bin-20261108'
FORMAL=RUN+'/dsrl-formal-20261108'
MODEL=ROOT+'/models/pi05-milk-bin-sft-20000'
CONTROL='/home/hello/.cache/evoworld-dsrl-ssh/%n'

def readiness(d):
 if d.get('pipeline_exit') is None:return 'waiting'
 checks={'pipeline_exit':0,'stage':'baseline_ready_for_review','latest_step':20000,'verified':True,
         'baseline_complete':True,'benchmark_valid':True,'total':20,'ready_for_dsrl':True}
 for k,v in checks.items():
  if d.get(k)!=v:raise ValueError(f'Gate rejected {k}: {d.get(k)!r}, expected {v!r}')
 if d['checkpoint']!=d['baseline_checkpoint']:raise ValueError('Baseline checkpoint differs from latest SFT')
 if d['seeds']!=list(range(20000,20020)):raise ValueError('Baseline seeds mismatch')
 if d['success_rate']!=d['successes']/20 or not .3<=d['success_rate']<=.7:raise ValueError('Baseline competence outside agreed30–70%')
 return 'ready'

def probe():
 import hashlib
 run=Path(RUN);exitfile=run/'pipeline-exit-status.txt'
 if not exitfile.exists():
  pid=int((run/'pipeline.pid').read_text());os.kill(pid,0)
  return {'pipeline_exit':None,'pipeline_pid':pid,'stage':(run/'pipeline-stage.txt').read_text().strip()}
 d={'pipeline_exit':int(exitfile.read_text()),'stage':(run/'pipeline-stage.txt').read_text().strip()}
 if d['pipeline_exit']!=0:return d
 candidates=list((run/'sft/milk-bin-pi05-sft/checkpoints').glob('global_step_*/actor/model_state_dict/full_weights.pt'))
 weight=max(candidates,key=lambda p:int(p.parents[2].name.removeprefix('global_step_')))
 baseline=json.loads((run/'baseline-calibration/report.json').read_text());gate=json.loads((run/'baseline-gate.json').read_text())
 d.update(latest_step=int(weight.parents[2].name.removeprefix('global_step_')),checkpoint=str(weight.parents[1].resolve()),
  verified=json.loads((run/'formal-checkpoint-verification.json').read_text())['passed'],baseline_checkpoint=baseline['checkpoint'],
  baseline_complete=baseline['complete'],benchmark_valid=baseline['benchmark_valid'],total=baseline['total'],
  seeds=[e['seed'] for e in baseline['episodes']],successes=baseline['successes'],success_rate=baseline['success_rate'],ready_for_dsrl=gate['ready_for_dsrl'])
 readiness(d)
 d['files']=[]
 for p,target in [(weight,MODEL+'/model_state_dict/full_weights.pt'),(run/'lerobot-train/norm_stats.json',MODEL+'/norm_stats.json')]:
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  d['files'].append({'source':str(p),'target':target,'bytes':p.stat().st_size,'sha256':digest})
 return d

def ssh_args(host,command):
 return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','-o',f'ControlPath={CONTROL}',host,command]

def remote(host,command,timeout=120):
 return subprocess.check_output(ssh_args(host,command),timeout=timeout,text=True)

def transfer(item):
 target=item['target'];temporary=target+'.handoff-part'
 command='mkdir -p '+shlex.quote(str(Path(target).parent))+' && cat > '+shlex.quote(temporary)
 with subprocess.Popen(ssh_args('5090-1','cat '+shlex.quote(item['source'])),stdout=subprocess.PIPE) as source:
  with subprocess.Popen(ssh_args('D46-6000x1',command),stdin=source.stdout) as dest:
   source.stdout.close()
   try:
    dest_code=dest.wait(timeout=1800);source_code=source.wait(timeout=60)
   except BaseException:
    source.kill();dest.kill();raise
   if dest_code or source_code:raise RuntimeError(f'Transfer failed source={source_code} destination={dest_code}')
 code="import hashlib,pathlib; p=pathlib.Path(%r); assert p.stat().st_size==%d; f=p.open('rb'); assert hashlib.file_digest(f,'sha256').hexdigest()==%r; f.close(); p.replace(%r)"%(temporary,item['bytes'],item['sha256'],target)
 remote('D46-6000x1','/root/miniconda3/bin/python -c '+shlex.quote(code),timeout=300)

def main():
 p=argparse.ArgumentParser();p.add_argument('--probe',action='store_true');p.add_argument('--state-dir');a=p.parse_args()
 if a.probe:print(json.dumps(probe()));return
 state=Path(a.state_dir);state.mkdir(parents=True,exist_ok=True)
 lock=(state/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def status(stage,**extra):
  value={'stage':stage,'pid':os.getpid(),'time':time.strftime('%Y-%m-%d %H:%M:%S'),**extra}
  tmp=state/'status.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(state/'status.json');print(json.dumps(value),flush=True)
 started=time.monotonic()
 try:
  if (state/'launched.json').exists():raise RuntimeError('Already launched; refusing duplicate')
  status('waiting_for_sft_and_baseline')
  while True:
   if time.monotonic()-started>6*3600:raise TimeoutError('SFT/baseline wait exceeded6hours')
   d=json.loads(remote('5090-1',ROOT+'/envs/rlinf-py312/bin/python '+ROOT+'/adapters/milk_bin/dsrl_handoff.py --probe',timeout=300))
   if readiness(d)=='ready':break
   status('waiting_for_sft_and_baseline',source=d);time.sleep(30)
  (state/'source-manifest.json').write_text(json.dumps(d,indent=2))
  remote('D46-6000x1','test ! -e '+FORMAL+'/started.json && /root/miniconda3/bin/python -c '+shlex.quote('import shutil; assert shutil.disk_usage("/root/autodl-tmp").free>100*2**30'))
  status('transferring_verified_sft',latest_step=d['latest_step'],success_rate=d['success_rate'])
  for item in d['files']:transfer(item)
  encoded=base64.b64encode(json.dumps(d,indent=2).encode()).decode()
  code=f'import pathlib,base64; p=pathlib.Path({FORMAL!r}); p.mkdir(parents=True,exist_ok=True); (p/"source-manifest.json").write_bytes(base64.b64decode({encoded!r}))'
  remote('D46-6000x1','/root/miniconda3/bin/python -c '+shlex.quote(code))
  command=f'cd {ROOT} && nohup bash adapters/milk_bin/run_dsrl_formal.sh > {FORMAL}/launcher.log 2>&1 < /dev/null &'
  remote('D46-6000x1',command)
  for attempt in range(36):
   time.sleep(5)
   launch_state=remote('D46-6000x1','if test -f '+FORMAL+'/exit-status.txt; then cat '+FORMAL+'/exit-status.txt >&2; exit 1; elif test -f '+FORMAL+'/started.json; then cat '+FORMAL+'/started.json; else echo pending; fi')
   if launch_state.strip()!='pending':
    launch=json.loads(launch_state);break
  else:raise TimeoutError('Formal launcher did not pass preflight within180seconds')
  (state/'launched.json').write_text(json.dumps(launch,indent=2));status('rl_launched',launch=launch)
 except BaseException as error:
  status('stopped',error=str(error));raise

if __name__=='__main__':
 raise SystemExit('Historical cross-host launcher disabled; use portable configuration and evoworld rl')
