"""Read-only capability checks: no network, inference, GPU contexts or training."""
import json,subprocess
from pathlib import Path
REQUIRED={
 'generate':(['gpt'],[],['openai','yaml']),
 'collect':(['expert','sim'],['expert_source','m2t2','isaac_assets','photo_scene','storage_box','office_background','env_template'],['torch','curobo','m2t2']),
 'convert':(['convert'],[],['lerobot','h5py']),
 'sft':(['rlinf'],['base_model','train_data'],['torch','rlinf','openpi']),
 'rl':(['rlinf','sim'],['checkpoint','train_data','expert_source','isaac_assets','photo_scene','storage_box','office_background','env_template'],['torch','rlinf']),
 'eval':(['rlinf','sim'],['checkpoint','train_data','expert_source','isaac_assets','photo_scene','storage_box','office_background','env_template'],['torch','rlinf'])}
def check(c):
 result={}
 for name,(envs,resources,modules) in REQUIRED.items():
  if name in c.get('disabled_components',{}):
   result[name]={'prerequisites_available':True,'disabled':True,'reason':c['disabled_components'][name],'execution_verified':False};continue
  missing=[f'environments.{k}' for k in envs if not Path(c['environments'].get(k,'/__missing__')).is_file()]
  missing += [f'resources.{k}' for k in resources if not Path(c['resources'].get(k,'/__missing__')).exists()]
  py=c['environments'].get(envs[0])
  if py and Path(py).is_file():
   code='import importlib.util,json; print(json.dumps({m:importlib.util.find_spec(m) is not None for m in '+repr(modules)+'}))'
   try:
    probe=subprocess.run([py,'-c',code],capture_output=True,text=True,timeout=30)
    if probe.returncode:missing.append('module-probe-failed')
    else:missing += ['module.'+m for m,ok in json.loads(probe.stdout).items() if not ok]
   except (OSError,ValueError,subprocess.TimeoutExpired):missing.append('module-probe-failed')
  result[name]={'prerequisites_available':not missing,'missing':missing,'execution_verified':False}
 return result
