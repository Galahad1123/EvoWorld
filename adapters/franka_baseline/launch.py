"""One bounded attempt, fresh output, preserved logs and actual exit/artifact checks."""
import datetime,hashlib,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
batch=Path(tempfile.mkdtemp(prefix='phase3-franka-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-',dir=root/'experiments'))
source=batch/'executed-source';source.mkdir()
for p in Path(__file__).parent.glob('*.py'):(source/p.name).write_bytes(p.read_bytes())
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')}
(batch/'source-hashes.json').write_text(json.dumps(hashes,indent=2))
env=os.environ.copy()
for k in list(env):
    if any(w in k.upper() for w in ['API_KEY','TOKEN','SECRET','PASSWORD']):env.pop(k,None)
env.update(OMNI_KIT_ACCEPT_EULA='YES',OMP_NUM_THREADS='8',SLURM_JOB_ID='evoworld-phase3',PYTHONUNBUFFERED='1')
start=time.monotonic();gpu=[];timeout=False
with (batch/'simulation.log').open('w') as log:
    proc=subprocess.Popen([str(root/'envs/isaacsim6/bin/python'),'-u',str(Path(__file__).parent/'run.py'),'--out',str(batch/'episode')],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(str(batch),flush=True)
    try:
        while proc.poll() is None:
            if time.monotonic()-start>300:timeout=True;break
            try:
                s=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
                gpu.append({'wall_seconds':time.monotonic()-start,'whole_gpu_memory_mib':int(s.stdout.strip().splitlines()[0])})
            except Exception:pass
            time.sleep(2)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
report_path=batch/'episode/report.json'
report=json.loads(report_path.read_text()) if report_path.exists() else {}
passed=(not timeout and proc.returncode==0 and report.get('passed') is True and not (batch/'episode/error.json').exists()
        and 'EVOWORLD_FRANKA_BASELINE_COMPLETE ' in (batch/'simulation.log').read_text(errors='replace'))
result={'passed':passed,'exit_code':proc.returncode,'timeout':timeout,'wall_seconds':time.monotonic()-start,'gpu_samples':gpu,
        'whole_gpu_peak_mib':max((s['whole_gpu_memory_mib'] for s in gpu),default=None),'episode_report':report,'new_gpt_calls':0}
(batch/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='gpu_samples'}),flush=True)
sys.exit(0 if passed else 1)
