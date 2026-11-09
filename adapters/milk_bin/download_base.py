"""Download the official Pi05 base checkpoint; verify each object's GCS CRC32C."""
import base64, concurrent.futures, hashlib, json, os
from pathlib import Path
from urllib.parse import quote
import requests
import google_crc32c
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'models/pi05-base-jax'; OUT.mkdir(parents=True,exist_ok=True)
PREFIX='checkpoints/pi05_base/'
PROXY={'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'}
r=requests.get('https://storage.googleapis.com/storage/v1/b/openpi-assets/o',params={'prefix':PREFIX,'maxResults':1000},proxies=PROXY,timeout=60);r.raise_for_status();manifest=r.json()
assert not manifest.get('nextPageToken')
(OUT/'gcs-manifest.json').write_text(json.dumps(manifest,indent=2))
def fetch(item):
 rel=item['name'][len(PREFIX):];dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 expected=item['crc32c']
 def digest(p):
  h=google_crc32c.Checksum()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  return base64.b64encode(h.digest()).decode()
 if dst.is_file() and dst.stat().st_size==int(item['size']) and digest(dst)==expected:return rel
 tmp=dst.with_name(dst.name+'.partial')
 url='https://storage.googleapis.com/openpi-assets/'+quote(item['name'],safe='/')+'?generation='+item['generation']
 with requests.get(url,proxies=PROXY,stream=True,timeout=(30,180)) as response:
  response.raise_for_status()
  with tmp.open('wb') as f:
   for chunk in response.iter_content(8*1024*1024):f.write(chunk)
 assert tmp.stat().st_size==int(item['size']) and digest(tmp)==expected,rel
 tmp.replace(dst);print('verified',rel,flush=True);return rel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results=list(pool.map(fetch,manifest['items']))
(OUT/'download-verified.json').write_text(json.dumps({'source':'gs://openpi-assets/checkpoints/pi05_base','files':results,'all_gcs_crc32c_verified':True},indent=2))
print('BASE_DOWNLOAD_COMPLETE',flush=True)
