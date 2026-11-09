"""Cache bounded official robot dependencies without launching Isaac or loading credentials."""
import hashlib,json,time,subprocess,os
from pathlib import Path,PurePosixPath
from pxr import Sdf,UsdUtils
ROOT=Path(__file__).resolve().parents[2]
BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/'
CACHE=ROOT/'assets/isaac6'
START='Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd'
started=time.monotonic();seen=set();pending={START};manifest=[];total=0

def fetch(rel):
    path=(CACHE/rel).resolve()
    if not path.is_relative_to(CACHE.resolve()):raise ValueError(rel)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.partial')
        env=os.environ.copy();env['HTTPS_PROXY']='http://127.0.0.1:7890'
        print('fetch '+rel,flush=True)
        seconds=max(1,min(30,int(180-(time.monotonic()-started))))
        remaining=min(50*1024*1024,200*1024*1024-total)
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--max-time',str(seconds),'--max-filesize',str(remaining),'--output',str(temporary),BASE+rel],env=env,check=True,timeout=seconds+1)
        temporary.replace(path)
    content=path.read_bytes()
    return rel,path,len(content),hashlib.sha256(content).hexdigest()

while pending:
    batch=sorted(pending-seen);pending=set()
    if len(seen)+len(batch)>200:raise RuntimeError('asset count budget exceeded')
    for rel in batch:
        if total>=200*1024*1024 or time.monotonic()-started>=180:raise RuntimeError('asset preparation budget exceeded')
        rel,path,size,sha=fetch(rel)
        seen.add(rel);total+=size
        if total>200*1024*1024:raise RuntimeError('asset total budget exceeded')
        manifest.append({'path':rel,'bytes':size,'sha256':sha,'url':BASE+rel})
        if path.suffix in ('.usd','.usda','.usdc'):
            for ref in set(r for group in UsdUtils.ExtractExternalReferences(str(path)) for r in group):
                if ref in ('OmniPBR.mdl','OmniGlass.mdl'):continue
                if ref.startswith(('http:','https:','omniverse:')):raise ValueError('unhandled external reference '+ref)
                dep=(path.parent/ref).resolve()
                if not dep.is_relative_to(CACHE.resolve()):raise ValueError(ref)
                pending.add(dep.relative_to(CACHE.resolve()).as_posix())
    pending-=seen
CACHE.mkdir(parents=True,exist_ok=True)
(CACHE/'franka-manifest.json').write_text(json.dumps({'files':manifest,'bytes':total,'wall_seconds':time.monotonic()-started},indent=2))
print(json.dumps({'files':len(manifest),'bytes':total,'wall_seconds':time.monotonic()-started}))
