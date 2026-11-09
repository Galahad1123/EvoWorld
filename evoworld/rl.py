"""Preserve the accepted baseline gate and four-hour DSRL execution boundary."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from .config import ROOT
from .resources import digest

def gate(c):
 p=Path(c['resources'].get('baseline_manifest','/__missing__'))
 if not p.is_file():raise ValueError('A verified baseline_manifest is required for formal RL')
 d=json.loads(p.read_text())
 sys.path.insert(0,str(ROOT/'adapters/milk_bin'))
 from dsrl_handoff import readiness
 if readiness(d)!='ready':raise ValueError('Baseline is not ready')
 checkpoint=Path(c['resources']['checkpoint'])/'model_state_dict/full_weights.pt'
 weights=[f for f in d.get('files',[]) if f.get('sha256')==digest(checkpoint)]
 if not weights:raise ValueError('Baseline manifest does not identify the selected checkpoint hash')
 return d

def run(c,cmd,env,output):
 gate(c)
 lock=Path(c['paths']['runs'])/'formal-rl.lock';lock.parent.mkdir(parents=True,exist_ok=True)
 with lock.open('w') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  marker=str(output/'audit');env['EVOWORLD_DSRL_AUDIT_DIR']=marker;env['EVOWORLD_DSRL_DEADLINE']=str(time.time()+14400)
  child=None
  try:
   child=subprocess.Popen(cmd,cwd=output,env=env,start_new_session=True)
   try:return child.wait(timeout=14400)
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
