"""Clean only processes bearing this run's exact audit-directory marker."""
import argparse,json,os,signal,time
from pathlib import Path

def matching(marker):
 needle=('EVOWORLD_DSRL_AUDIT_DIR='+marker).encode()
 excluded=set();pid=os.getpid()
 while pid>0:
  excluded.add(pid)
  try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
  except (OSError,ValueError,IndexError):break
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name) in excluded:continue
  try:
   if needle in (p/'environ').read_bytes().split(b'\0'):found.append(int(p.name))
  except (OSError,ProcessLookupError):pass
 return found

def cleanup(marker,grace=5):
 if not marker:raise ValueError('A unique run marker is required')
 initial=matching(marker)
 for pid in initial:
  try:os.kill(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 deadline=time.monotonic()+grace
 while matching(marker) and time.monotonic()<deadline:time.sleep(.1)
 forced=matching(marker)
 for pid in forced:
  try:os.kill(pid,signal.SIGKILL)
  except ProcessLookupError:pass
 for _ in range(30):
  if not matching(marker):break
  time.sleep(.1)
 return {'matched':initial,'forced':forced,'remaining':matching(marker)}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--marker',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 result=cleanup(a.marker);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
 if result['remaining']:raise SystemExit(1)
