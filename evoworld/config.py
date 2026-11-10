"""One resolver shared by command-line, Python and shell entrypoints."""
import os
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
SECRET_KEYS={'OPENAI_API_KEY'}
def path(value,base):
 p=Path(value).expanduser()
 return os.path.abspath(p if p.is_absolute() else base/p)
def load(filename=None):
 p=Path(filename or os.environ.get('EVOWORLD_CONFIG',ROOT/'configs/local.yaml')).expanduser().resolve()
 if not p.is_file():raise FileNotFoundError(f'Create machine configuration from configs/local.example.yaml: {p}')
 c=yaml.safe_load(p.read_text()) or {};base=p.parent
 storage=path(c.get('storage_root','../storage'),base)
 c['storage_root']=storage;c['config_file']=str(p);c['repo_root']=str(ROOT)
 c['paths']={k:path(c.get('paths',{}).get(k,str(Path(storage)/k)),base) for k in ('models','datasets','assets','runs','cache')}
 for group in ('environments','upstreams','resources'):
  c[group]={k:path(v,base) for k,v in c.get(group,{}).items()}
 c['research']={**c.get('research',{}),'output_root':path(c.get('research',{}).get('output_root',str(ROOT/'research/rl/outputs')),base)}
 c['runtime_library_dirs']=[path(v,base) for v in c.get('runtime_library_dirs',[])]
 if c.get('isaac_experience'):c['isaac_experience']=path(c['isaac_experience'],base)
 c['credentials_file']=path(os.environ.get('EVOWORLD_CREDENTIALS_FILE',c.get('credentials_file','~/.config/evoworld/credentials.env')),base)
 return c
def credentials(filename,environ=None):
 # No interpolation, execution, or arbitrary env propagation.
 result={};p=Path(filename).expanduser()
 if p.is_file():
  for line in p.read_text().splitlines():
   line=line.strip()
   if not line or line.startswith('#'):continue
   k,sep,v=line.partition('=');k=k.strip()
   if sep and k in SECRET_KEYS:result[k]=v.strip().strip('\"\'')
 source=os.environ if environ is None else environ
 result.update({k:source[k] for k in SECRET_KEYS if source.get(k)})
 return result
def resource(name):
 c=load()
 if name not in c['resources']:raise KeyError(f'Missing resource mapping: {name}')
 p=Path(c['resources'][name])
 if not p.exists():raise FileNotFoundError(f'Missing resource {name}; see docs/RESOURCES.md')
 return p
