"""Hash-bound baseline admission and a configurable bounded DSRL run."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from .config import ROOT
from .resources import digest

def gate(c):
 p=Path(c['resources'].get('baseline_manifest','/__missing__'))
 if not p.is_file():raise ValueError('A verified baseline_manifest is required for formal RL')
 d=json.loads(p.read_text())
 accepted=c.get('rl',{}).get('accepted_baseline')
 if accepted is not None:
  if not accepted.get('reason'):raise ValueError('Explicit baseline acceptance reason is required')
  if digest(p)!=accepted.get('report_sha256'):raise ValueError('Accepted baseline report changed')
  if d.get('complete') is not True or d.get('total')!=20:raise ValueError('Complete 20-seed baseline required')
  rows=d.get('episodes',[])
  if len(rows)!=20 or [r.get('seed') for r in rows]!=list(range(20000,20020)):raise ValueError('Baseline seeds mismatch')
  if any(type(r.get('success')) is not bool for r in rows):raise ValueError('Invalid episode success flag')
  successes=sum(r['success'] for r in rows)
  if d.get('successes')!=successes or d.get('success_rate')!=successes/20:raise ValueError('Inconsistent baseline success rate')
  for key in ('success_rate','max_control_steps','executed_chunk'):
   if d.get(key)!=accepted.get(key):raise ValueError('Accepted baseline mismatch: '+key)
  horizon,chunk=d['max_control_steps'],d['executed_chunk']
  if type(horizon) is not int or type(chunk) is not int or not 1<=chunk<=20 or horizon<=0 or horizon%chunk:raise ValueError('Invalid execution profile')
  checkpoint=Path(c['resources']['checkpoint'])
  if Path(d['checkpoint']).resolve()!=checkpoint.resolve():raise ValueError('Baseline checkpoint path mismatch')
  if digest(checkpoint/'model_state_dict/full_weights.pt')!=accepted.get('checkpoint_sha256'):raise ValueError('Accepted checkpoint changed')
  return d
 sys.path.insert(0,str(ROOT/'adapters/milk_bin'))
 from dsrl_handoff import readiness
 if readiness(d)!='ready':raise ValueError('Baseline is not ready')
 checkpoint=Path(c['resources']['checkpoint'])/'model_state_dict/full_weights.pt'
 weights=[f for f in d.get('files',[]) if f.get('sha256')==digest(checkpoint)]
 if not weights:raise ValueError('Baseline manifest does not identify the selected checkpoint hash')
 return d

def run(c,cmd,env,output):
 admission=gate(c)
 seconds=c.get('rl',{}).get('walltime_seconds',14400)
 if type(seconds) is not int or seconds<=300:raise ValueError('RL walltime_seconds must be an integer greater than the 300-second save reserve')
 cmd=list(cmd);accepted=c.get('rl',{}).get('accepted_baseline')
 if accepted is not None:
  chunk=accepted['executed_chunk'];horizon=accepted['max_control_steps']
  cmd += [f'actor.model.num_action_chunks={chunk}',f'++actor.model.openpi.action_chunk={chunk}']
  cmd += [f'env.{split}.{key}={horizon}' for split in ('train','eval') for key in ('max_episode_steps','max_steps_per_rollout_epoch')]
 lock=Path(c['paths']['runs'])/'formal-rl.lock';lock.parent.mkdir(parents=True,exist_ok=True)
 with lock.open('w') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  marker=str(output/'audit');env['EVOWORLD_DSRL_AUDIT_DIR']=marker;env['EVOWORLD_DSRL_DEADLINE']=str(time.time()+seconds)
  (output/'rl-launch.json').write_text(json.dumps({'command':cmd,'walltime_seconds':seconds,'deadline_unix':float(env['EVOWORLD_DSRL_DEADLINE']),'accepted_baseline':accepted},indent=2,ensure_ascii=False))
  (output/'admission.json').write_text(json.dumps(admission,indent=2))
  child=None
  try:
   child=subprocess.Popen(cmd,cwd=output,env=env,start_new_session=True)
   try:return child.wait(timeout=seconds)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGUSR1)
    try:return child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait();return 124
  finally:
   if child and child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   sys.path.insert(0,str(ROOT/'adapters/milk_bin'))
   from dsrl_cleanup import cleanup
   result=cleanup(marker);(output/'cleanup.json').write_text(json.dumps(result,indent=2))
   if result['remaining']:raise RuntimeError('Owned RL subprocess cleanup failed')
