"""Pinned, resumable file downloads with content checks; no second HF cache copy."""
import argparse,concurrent.futures,hashlib,json,os,time
from pathlib import Path
from urllib.parse import quote
import requests
import threading
LOCAL=threading.local()
def session():
 if not hasattr(LOCAL,"session"):LOCAL.session=requests.Session()
 return LOCAL.session
from huggingface_hub import get_token

def headers():
 token=get_token()
 return {'Authorization':'Bearer '+token} if token else {}

def get(url):
 for n in range(6):
  try:
   r=session().get(url,headers=headers(),timeout=(20,90));r.raise_for_status();return r
  except requests.RequestException:
   if n==5:raise
   time.sleep(min(2**n,20))

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--type',choices=['dataset','model'],required=True);p.add_argument('--dest',type=Path,required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
 host=os.environ.get('HF_ENDPOINT','https://hf-mirror.com').rstrip('/');api='datasets' if a.type=='dataset' else 'models'
 a.dest.mkdir(parents=True,exist_ok=True);manifest=a.dest/'download-manifest.json'
 if manifest.exists():m=json.loads(manifest.read_text());assert m['repo']==a.repo
 else:
  info=get(f'{host}/api/{api}/{a.repo}').json();rev=info['sha'];rows=[];url=f'{host}/api/{api}/{a.repo}/tree/{rev}?recursive=true&limit=1000'
  while url:
   r=get(url);rows.extend(x for x in r.json() if x['type']=='file');url=r.links.get('next',{}).get('url');print('metadata files',len(rows),flush=True)
  if a.type=='model':rows=[x for x in rows if x['path'].endswith(('.json','.safetensors','.model','.txt','.md'))]
  m={'repo':a.repo,'type':a.type,'revision':rev,'files':rows};manifest.write_text(json.dumps(m,indent=2))
 def valid(path,row):
  if not path.is_file() or path.stat().st_size!=row['size']:return False
  expected=row.get('lfs',{}).get('oid')
  if expected:
   with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()==expected
  h=hashlib.sha1();h.update(f"blob {row['size']}\0".encode())
  with path.open('rb') as f:
   for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
  return h.hexdigest()==row['oid']
 def download(row):
  rel=row['path'];out=a.dest/rel
  if valid(out,row):return row['size']
  out.parent.mkdir(parents=True,exist_ok=True);part=out.with_suffix(out.suffix+'.partial')
  base=f"{host}/{'datasets/' if a.type=='dataset' else ''}{a.repo}/resolve/{m['revision']}/{quote(rel)}"
  for attempt in range(8):
   try:
    with session().get(base+f'?download=true&attempt={attempt}',headers=headers(),stream=True,timeout=(30,120)) as r:
     r.raise_for_status()
     with part.open('wb') as f:
      for data in r.iter_content(1024*1024):f.write(data)
    if not valid(part,row):raise ValueError('size/hash mismatch: '+rel)
    part.replace(out);return row['size']
   except Exception as e:
    if attempt==7:raise
    print('retry',rel,type(e).__name__,flush=True);time.sleep(min(2**attempt,20))
 def priority(row):
  path=row['path']
  if '/meta/' in path:return (0,path)
  if path.startswith('libero10_task0_sft/'):return (1,path)
  if 'episode_000000.' in path or 'episode_000001.' in path:return (2,path)
  if path.startswith('libero10_task0_eval/'):return (3,path)
  return (4,path)
 m['files'].sort(key=priority)
 start=time.time();done=0;total=sum(x['size'] for x in m['files']);print('START',m['revision'],len(m['files']),total,flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for i,size in enumerate(pool.map(download,m['files']),1):
   done+=size
   if i%50==0 or i==len(m['files']):
    status={'files':i,'total_files':len(m['files']),'bytes':done,'total_bytes':total,'seconds':time.time()-start};(a.dest/'download-progress.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
 (a.dest/'verified.json').write_text(json.dumps({'revision':m['revision'],'files':len(m['files']),'bytes':total,'verified':True},indent=2))
if __name__=='__main__':main()
