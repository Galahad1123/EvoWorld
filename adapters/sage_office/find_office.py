"""Bounded metadata-only search of official SAGE archives using HTTP ranges."""
import concurrent.futures,io,json,random,zipfile
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'assets/SAGE-10k/office-search';OUT.mkdir(exist_ok=True)

class RemoteZip(io.RawIOBase):
 def __init__(self,item):
  self.size=item['size'];self.pos=0;self.cache=[]
  self.url='https://hf-mirror.com/datasets/nvidia/SAGE-10k/resolve/main/'+item['path']
  self.session=requests.Session();self.session.trust_env=False
  self.session.proxies={'https':'http://127.0.0.1:7890'}
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  end=min(self.size,self.pos+n)
  if end<=self.pos:return b''
  for a,b,data in self.cache:
   if a<=self.pos and b>=end:
    result=data[self.pos-a:end-a];self.pos=end;return result
  start=self.pos;fetch_end=min(self.size,max(end,start+65536))
  with self.session.get(self.url+'?download=true&offset='+str(start),headers={'Range':f'bytes={start}-{fetch_end-1}'},timeout=(10,25),stream=True) as r:
   r.raise_for_status()
   if r.status_code!=206:raise RuntimeError('Server ignored Range; avoiding full archive download')
   assert r.headers['Content-Range'].startswith(f'bytes {start}-')
   data=r.content
  assert len(data)==fetch_end-start
  self.cache.append((start,fetch_end,data));self.pos=end;return data[:end-start]

def inspect(item):
 try:
  remote=RemoteZip(item)
  with zipfile.ZipFile(remote) as z:
   name=next(n for n in z.namelist() if n.endswith('.json') and '/' not in n)
   layout=json.loads(z.read(name));rooms=layout['rooms']
   result={'path':item['path'],'size':item['size'],'layout_id':layout['id'],'room_types':[r['room_type'] for r in rooms],'building_style':layout.get('building_style',''),'metadata_bytes_fetched':sum(len(x[2]) for x in remote.cache)}
   if any('office' in r['room_type'].lower() for r in rooms):
    dest=OUT/layout['id'];dest.mkdir(exist_ok=True)
    (dest/name).write_text(json.dumps(layout,indent=2))
    for n in z.namelist():
     if n.startswith('preview/') and n.endswith('.png'):
      (dest/Path(n).name).write_bytes(z.read(n))
    result['office_found']=True
   return result
 except Exception as e:return {'path':item['path'],'error':str(e)}

items=json.loads((ROOT/'assets/SAGE-10k/office-candidate-index.json').read_text())
random.Random(408).shuffle(items)
results=json.loads((OUT/'search.json').read_text()) if (OUT/'search.json').exists() else []
completed={r['path'] for r in results};items=[i for i in items if i['path'] not in completed]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for start in range(0,30,3):
  batch=list(pool.map(inspect,items[start:start+3]))
  results+=batch
  for row in batch:print(json.dumps(row),flush=True)
  (OUT/'search.json').write_text(json.dumps(results,indent=2))
  if sum(bool(r.get('office_found')) for r in results)>=3:break
assert any(r.get('office_found') for r in results),'No office in bounded metadata sample'
