import subprocess
from .config import load

def isaac_experience(config=None):
 c=config if config is not None else load()
 if c.get('isaac_experience'):
  from pathlib import Path
  if not Path(c['isaac_experience']).is_file():raise FileNotFoundError('Configured Isaac experience missing')
  return c['isaac_experience']
 code="import importlib.util,pathlib; s=importlib.util.find_spec('isaacsim'); print(pathlib.Path(next(iter(s.submodule_search_locations)))/'apps/isaacsim.exp.base.python.kit')"
 p=subprocess.check_output([c['environments']['sim'],'-c',code],text=True).strip()
 from pathlib import Path
 if not Path(p).is_file():raise FileNotFoundError('Isaac Sim experience not found in configured environment')
 return p

def runtime_environment(c,environ):
 """只为当前子进程添加机器配置中的动态库目录。"""
 import os
 from pathlib import Path
 e=dict(environ);dirs=c.get('runtime_library_dirs',[])
 for d in dirs:
  if not Path(d).is_dir():raise FileNotFoundError(f'Runtime library directory missing: {d}')
 if dirs:e['LD_LIBRARY_PATH']=os.pathsep.join(dirs+([e['LD_LIBRARY_PATH']] if e.get('LD_LIBRARY_PATH') else []))
 return e
