"""Lightweight dispatch. Never install dependencies or download weights implicitly."""
import argparse,json,os,subprocess,sys,datetime
from pathlib import Path
from .config import ROOT,load,credentials
COMMANDS={
 'generate':('gpt','adapters/realtosim_photo/agent.py'),
 'collect':('expert','adapters/milk_bin/collect.py'),
 'convert':('convert','adapters/milk_bin/export_dataset.py'),
 'sft':('rlinf','@RLinf/examples/sft/train_vla_sft.py'),
 'rl':('rlinf','adapters/milk_bin/dsrl_formal_entry.py'),
 'eval':('rlinf','adapters/milk_bin/evaluate.py')}
def command(c,name,args):
 component,script=COMMANDS[name]
 python=c['environments'].get(component)
 if not python:raise ValueError(f'Missing environments.{component}')
 if script.startswith('@'):
  upstream,rel=script[1:].split('/',1);script=str(Path(c['upstreams'][upstream])/rel)
 else:script=str(ROOT/script)
 return [python,'-u',script,*args]
def main():
 p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('action',choices=['doctor',*COMMANDS]);p.add_argument('--dry-run',action='store_true');p.add_argument('--output');p.add_argument('--component',choices=list(COMMANDS))
 a,args=p.parse_known_args();args=args[1:] if args[:1]==['--'] else args
 c=load(a.config)
 if a.action=='doctor':
  from .doctor import check
  report=check(c)
  if a.component:report={a.component:report[a.component]}
  print(json.dumps(report,indent=2,ensure_ascii=False));return int(any(not x['prerequisites_available'] for x in report.values()))
 if a.action in c.get('disabled_components',{}):raise ValueError(c['disabled_components'][a.action])
 if a.action in ('sft','rl') and not any(x.startswith('--config') for x in args):
  args=['--config-path',str(ROOT/'adapters/milk_bin'),'--config-name','sft' if a.action=='sft' else 'dsrl_formal',*args]
 # Resolve user paths before changing the subprocess working directory.
 if a.action=='generate':
  for i in [0,2,*range(3,len(args))]:
   if i<len(args):args[i]=str(Path(args[i]).expanduser().resolve())
 else:
  path_flags={'--checkpoint','--dataset','--output','--run','--out'}
  for i in range(1,len(args)):
   if args[i-1] in path_flags:args[i]=str(Path(args[i]).expanduser().resolve())
 cmd=command(c,a.action,args)
 if a.dry_run:print(json.dumps({'command':cmd,'component':COMMANDS[a.action][0],'resources':c['resources']},indent=2));return 0
 from .resources import verify,compatibility
 compatibility(c,a.action)
 evidence=verify(c)
 if not Path(cmd[0]).is_file():raise FileNotFoundError('Configured interpreter is missing; run doctor')
 output=Path(a.output).expanduser().resolve() if a.output else Path(c['paths']['runs'])/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-'+a.action)
 output.mkdir(parents=True,exist_ok=False)
 env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('API_KEY','TOKEN','PASSWORD','SECRET'))}
 env['EVOWORLD_CONFIG']=c['config_file'];env['EVOWORLD_ROOT']=str(ROOT)
 env['PYTHONPATH']=os.pathsep.join([str(ROOT),str(ROOT/'adapters/milk_bin'),*c['upstreams'].values()])
 env['HF_HOME']=str(Path(c['paths']['cache'])/'huggingface');env['OPENPI_DATA_HOME']=str(Path(c['paths']['cache'])/'openpi')
 env['EVOWORLD_RUN_DIR']=str(output)
 for key,name in [('EVOWORLD_BASE_MODEL','base_model'),('EVOWORLD_TRAIN_DATA','train_data'),('EVOWORLD_CHECKPOINT','checkpoint')]:
  if c['resources'].get(name):env[key]=c['resources'][name]
 env['EVOWORLD_RLINF']=c['upstreams'].get('RLinf','')
 if a.action in ('collect','eval','rl'):
  from .runtime import isaac_experience,runtime_environment
  env=runtime_environment(c,env)
  env['EVOWORLD_ISAAC_EXPERIENCE']=isaac_experience(c)
  env['EVOWORLD_ISAAC_KIT_ARGS']='--/isaaclab/cameras_enabled=true'
  if c['resources'].get('isaac_assets'):env['EVOWORLD_SAGE_ASSET_ROOT']=c['resources']['isaac_assets']
  if c['resources'].get('expert_source'):
   source=Path(c['resources']['expert_source'])
   extra=[str(source/'IsaacLab/source/extensions'/n) for n in ('omni.isaac.lab','omni.isaac.lab_assets','omni.isaac.lab_tasks')]
   env['PYTHONPATH']=os.pathsep.join(extra+[env['PYTHONPATH']])
 if a.action=='collect':
  from .expert import compiler_environment
  env=compiler_environment(c,env)
  env.update(EVOWORLD_SAGE_FRANKA_ONLY='1',EVOWORLD_TARGET_OBJECT='milk',EVOWORLD_PLACE_OBJECT='desk',EVOWORLD_TABLE_OBJECT='desk')
 env['PYTHONNOUSERSITE']='1'
 env['JAX_PLATFORMS']='cpu';env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
 env['EMBODIED_PATH']=str(Path(c['upstreams'].get('RLinf',''))/'examples/embodiment')
 env['REPO_PATH']=c['upstreams'].get('RLinf','')
 net=c.get('network',{})
 for key,field in [('HF_ENDPOINT','hf_endpoint'),('PIP_INDEX_URL','pip_index_url')]:
  if net.get(field):env[key]=net[field]
 if a.action=='generate':
  env.update(credentials(c['credentials_file']))
  if not env.get('OPENAI_API_KEY') or env['OPENAI_API_KEY'] in ('YOUR_API_KEY','你的实际密钥'):raise ValueError('Set OPENAI_API_KEY in the configured credentials file or environment')
  env['OPENAI_BASE_URL']=net.get('gpt_base_url','https://api.openai.com/v1')
  if net.get('gpt_proxy'):
   from urllib.parse import urlsplit
   import socket
   u=urlsplit(net['gpt_proxy'])
   if u.scheme not in ('http','https') or not u.hostname:raise ValueError('Invalid GPT proxy')
   with socket.create_connection((u.hostname,u.port or (443 if u.scheme=='https' else 80)),timeout=3):pass
   env['HTTPS_PROXY']=env['https_proxy']=net['gpt_proxy']
 else:
  for key in list(env):
   if key.lower() in ('http_proxy','https_proxy','all_proxy'):env.pop(key)
  env['NO_PROXY']='localhost,127.0.0.1,::1'
  env['RLINF_EXT_MODULE']='dsrl_extension' if a.action=='rl' else 'rlinf_extension'
 commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
 # Capture public task settings and dependency identities; never machine secrets.
 import shutil
 shutil.copyfile(ROOT/'manifests/upstreams.json',output/'upstreams.json')
 if a.action=='generate':shutil.copyfile(ROOT/'configs/gpt.yaml',output/'gpt.yaml')
 if a.action in ('sft','rl','eval'):
  task_file=ROOT/'adapters/milk_bin'/('dsrl_formal.yaml' if a.action=='rl' else 'sft.yaml')
  shutil.copyfile(task_file,output/'task-template.yaml')
 # Do not serialize arbitrary inherited environment or credential/proxy endpoints.
 (output/'run.json').write_text(json.dumps({'command':cmd,'commit':commit,'resources':c['resources'],'action':a.action,'resource_evidence':evidence,'observation_contract':c.get('observation_contract'),'dirty':bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True))},indent=2))
 if a.action=='rl':
  from .rl import run
  return run(c,cmd,env,output)
 return subprocess.call(cmd,cwd=output,env=env)
if __name__=='__main__':sys.exit(main())
