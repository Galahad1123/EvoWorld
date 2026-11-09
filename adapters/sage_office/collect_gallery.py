"""Collect office metadata and official previews without downloading scene assets."""
import argparse,concurrent.futures,hashlib,io,json,random,struct,time,zipfile,zlib
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
HOST='https://hf-mirror.com'

def session():
 s=requests.Session();s.trust_env=False;s.proxies={'https':'http://127.0.0.1:7890'};return s

class RemoteZip(io.RawIOBase):
 def __init__(self,item,revision):
  self.size=item['size'];self.pos=0;self.cache=[];self.transferred=0;self.session=session()
  self.url=f'{HOST}/datasets/nvidia/SAGE-10k/resolve/{revision}/'+item['path']
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if self.pos<0:raise ValueError('Negative seek')
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  end=min(self.size,self.pos+n)
  if end<=self.pos:return b''
  for start,stop,data in self.cache:
   if start<=self.pos and stop>=end:
    result=data[self.pos-start:end-start];self.pos=end;return result
  start=self.pos;stop=min(self.size,max(end,start+65536))
  if stop-start>12*1024*1024:raise ValueError('Requested member exceeds preview budget')
  error=None
  for attempt in range(2):
   try:
    with self.session.get(self.url+f'?gallery_range={start}-{stop-1}',headers={'Range':f'bytes={start}-{stop-1}'},timeout=(10,25),stream=True) as r:
     r.raise_for_status()
     if r.status_code!=206:raise RuntimeError('Range unsupported; full archive download refused')
     assert r.headers['Content-Range']==f'bytes {start}-{stop-1}/{self.size}'
     chunks=[];received=0
     for chunk in r.iter_content(65536):
      received+=len(chunk);self.transferred+=len(chunk)
      if received>stop-start:raise RuntimeError('Range body exceeded requested bytes')
      chunks.append(chunk)
     data=b''.join(chunks);assert len(data)==stop-start
    self.cache.append((start,stop,data));self.pos=end;return data[:end-start]
   except Exception as exc:error=exc
  raise error
 def prefix_layout(self):
  # The official archives put the layout before object meshes. Validate the ZIP
  # header/CRC and fall back to central-directory lookup for other layouts.
  self.seek(0);data=self.read(65536);offset=0
  while offset+30<=len(data):
   h=struct.unpack_from('<IHHHHHIIIHH',data,offset)
   if h[0]!=0x04034b50 or h[2]&9:break
   start=offset+30+h[9]+h[10];name=data[offset+30:offset+30+h[9]].decode('utf-8')
   if name.endswith('.json') and '/' not in name:
    if h[8]>4*1024*1024:raise ValueError('Layout JSON too large')
    if start+h[7]>len(data):self.seek(start);compressed=self.read(h[7])
    else:compressed=data[start:start+h[7]]
    raw=zlib.decompress(compressed,-15) if h[3]==8 else compressed
    assert len(raw)==h[8] and zlib.crc32(raw)==h[6]
    return json.loads(raw)
   offset=start+h[7]
  with zipfile.ZipFile(self) as z:
   name=next(n for n in z.namelist() if n.endswith('.json') and '/' not in n)
   return json.loads(z.read(name))

def inspect(item,revision,cache):
 remote=RemoteZip(item,revision);start=time.monotonic()
 try:
  data=remote.prefix_layout();rooms=data['rooms']
  result={'path':item['path'],'archive_bytes':item['size'],'archive_sha256':item.get('lfs',{}).get('oid'),'layout_id':data['id'],'room_types':[r['room_type'] for r in rooms],'building_style':data.get('building_style','')}
  office=any('office' in r['room_type'].lower() for r in rooms)
  if office:
   dest=cache/data['id'];dest.mkdir(exist_ok=True)
   (dest/(data['id']+'.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2))
   previews=[]
   with zipfile.ZipFile(remote) as z:
    names=sorted(n for n in z.namelist() if n.startswith('preview/') and n.lower().endswith(('.png','.jpg','.jpeg')))
    for name in names[:2]:
     assert z.getinfo(name).file_size<12*1024*1024
     raw=z.read(name);target=dest/Path(name).name;target.write_bytes(raw)
     previews.append({'file':str(target),'sha256':hashlib.sha256(raw).hexdigest()})
   assert len(previews)==2,'Need two official previews'
   result.update(office_found=True,previews=previews)
  result.update(transferred_bytes=remote.transferred,seconds=time.monotonic()-start)
  return result
 except Exception as exc:return {'path':item['path'],'error':str(exc),'transferred_bytes':remote.transferred,'seconds':time.monotonic()-start}
 finally:remote.session.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--target',type=int,default=16);p.add_argument('--workers',type=int,default=6);p.add_argument('--max-scan',type=int,default=1200);a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);cache=ROOT/'assets/SAGE-10k/gallery-cache';cache.mkdir(exist_ok=True)
 s=session();r=s.get(HOST+'/api/datasets/nvidia/SAGE-10k',timeout=(10,30));r.raise_for_status();revision=r.json()['sha']
 index=[];url=f'{HOST}/api/datasets/nvidia/SAGE-10k/tree/{revision}/scenes?limit=1000'
 while url and len(index)<a.max_scan:
  r=s.get(url,timeout=(10,30));r.raise_for_status();index.extend(x for x in r.json() if x['path'].endswith('.zip'))
  url=r.links.get('next',{}).get('url')
  if url:url=url.replace('https://huggingface.co',HOST)
 (out/'index.json').write_text(json.dumps({'revision':revision,'items':index},indent=2))
 legacy=json.loads((ROOT/'assets/SAGE-10k/office-search/search.json').read_text());seen={x['path'] for x in legacy if 'error' not in x}
 bypath={x['path']:x for x in index};found=[]
 for row in legacy:
  if row.get('office_found'):
   dest=ROOT/'assets/SAGE-10k/office-search'/row['layout_id'];files=sorted(dest.glob('preview_*.png'))
   if len(files)>=2:
    found.append({**row,'cache_origin':'previous official archive range extraction','archive_sha256':bypath.get(row['path'],{}).get('lfs',{}).get('oid'),'previews':[{'file':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files[:2]]})
 candidates=[x for x in index if x['path'] not in seen];random.Random(20261108).shuffle(candidates)
 results=[];start=time.monotonic();(out/'collect_gallery-executed.py').write_bytes(Path(__file__).read_bytes())
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for offset in range(0,min(a.max_scan,len(candidates)),a.workers):
   batch=list(pool.map(lambda item:inspect(item,revision,cache),candidates[offset:offset+a.workers]));results+=batch
   for row in batch:
    if row.get('office_found'):
     found.append(row);print('FOUND '+json.dumps({'count':len(found),'layout_id':row['layout_id'],'style':row['building_style']},ensure_ascii=False),flush=True)
   summary={'complete':len(found)>=a.target,'dataset_revision':revision,'target':a.target,'cached_metadata_count':len(legacy),'new_scanned':len(results),'office_count':len(found),'transferred_bytes':sum(r.get('transferred_bytes',0) for r in results),'wall_seconds':time.monotonic()-start,'found':found,'scans':results}
   (out/'collection.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
   print(f"PROGRESS scanned={len(results)} offices={len(found)} MiB={summary['transferred_bytes']/1024**2:.1f}",flush=True)
   if len(found)>=a.target:break
 assert len(found)>=10,'Fewer than10 candidates; collection remains incomplete'
 print('COLLECTION_DONE '+json.dumps({k:v for k,v in summary.items() if k not in ['found','scans']}),flush=True)

if __name__=='__main__':main()
