"""Explicit resource integrity and policy observation compatibility checks."""
import hashlib,json
from pathlib import Path

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def verify(c):
 evidence={}
 for name,item in c.get('resource_checks',{}).items():
  root=Path(c['resources'][name]);files={}
  for rel,expected in item.get('sha256',{}).items():
   p=root/rel if root.is_dir() else root
   if root.is_dir() and not p.resolve().is_relative_to(root.resolve()):raise ValueError('Resource check escapes root')
   actual=digest(p)
   if actual!=expected:raise ValueError(f'Resource checksum mismatch: {name}/{rel}')
   files[rel]=actual
  evidence[name]={'version':item.get('version'),'sha256':files}
 return evidence

def compatibility(c,action):
 if action not in ('sft','rl','eval'):return
 required=['train_data']+(['checkpoint'] if action in ('rl','eval') else [])
 expected=c.get('observation_contract')
 if not expected:raise ValueError('Set observation_contract before learning/evaluation')
 for name in required:
  found=c.get('resource_checks',{}).get(name,{}).get('observation_contract')
  if found!=expected:raise ValueError(f'Observation/action contract mismatch or missing: {name}')
