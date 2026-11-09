"""Bounded native SFT benchmark, with per-process GPU memory monitoring."""
import json,os,re,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'experiments/phase8-milk-bin-20261108'
def memory():
 s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
 return {int(p):int(m) for line in s.splitlines() if ',' in line for p,m in [line.split(',')]}
existing=set(memory());env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'adapters/milk_bin')+':'+env.get('PYTHONPATH','');env['RLINF_EXT_MODULE']='rlinf_extension';env['EVOWORLD_MODEL_GPU_FRACTION']='0.60'
for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env.pop(k,None)
env['NO_PROXY']='localhost,127.0.0.1'
cmd=[str(ROOT/'envs/rlinf-py312/bin/python'),'-u',str(ROOT/'upstream/RLinf/examples/sft/train_vla_sft.py'),'--config-path',str(ROOT/'adapters/milk_bin'),'--config-name','sft','runner.max_steps=10','runner.save_interval=-1','runner.logger.log_path='+str(RUN/'native-sft-batch4'),'actor.global_batch_size=4','actor.micro_batch_size=4','data.num_workers=0','data.train_data_paths='+str(RUN/'smoke-dataset2'),'actor.model.openpi_data.norm_stats_path='+str(RUN/'smoke-dataset2/norm_stats.json')]
peak=0;start=time.monotonic();samples=[]
with (RUN/'native-sft-batch4.log').open('w') as log:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
 while proc.poll() is None:
  used={pid:m for pid,m in memory().items() if pid not in existing};peak=max(peak,sum(used.values()));samples.append({'elapsed':time.monotonic()-start,'process_memory_mib':used})
  if time.monotonic()-start>600:proc.terminate();raise TimeoutError('SFT benchmark exceeded600s')
  time.sleep(.5)
assert proc.returncode==0,proc.returncode
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
files=list((RUN/'native-sft-batch4').rglob('events.out.tfevents.*'));assert files
values=[];losses=[]
for path in files:
 e=EventAccumulator(str(path));e.Reload()
 if 'time/training' in e.Tags()['scalars']:values.extend(v.value for v in e.Scalars('time/training'))
 if 'train/loss' in e.Tags()['scalars']:losses.extend(v.value for v in e.Scalars('train/loss'))
assert len(values)==10 and len(losses)==10
report={'passed':True,'optimizer_updates':10,'global_batch_size':4,'micro_batch_size':4,'gradient_accumulation':1,'mean_training_step_seconds_after_first':sum(values[1:])/len(values[1:]),'peak_process_gpu_memory_mib':peak,'losses':losses,'training_step_seconds':values,'wall_seconds':time.monotonic()-start}
(RUN/'native-sft-batch4-report.json').write_text(json.dumps(report,indent=2));(RUN/'native-sft-batch4-memory.json').write_text(json.dumps(samples));print(json.dumps(report),flush=True)
