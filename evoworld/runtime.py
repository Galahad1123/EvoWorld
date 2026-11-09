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
