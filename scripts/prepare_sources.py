"""Explicit source preparation, pinned commits and patches; no Python installation."""
import argparse,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(args):subprocess.run(args,check=True)
def patch(dest,path):
 if subprocess.run(['git','-C',str(dest),'apply','--reverse','--check',str(path)],capture_output=True).returncode==0:return
 run(['git','-C',str(dest),'apply','--check',str(path)])
 run(['git','-C',str(dest),'apply',str(path)])
def main():
 p=argparse.ArgumentParser();p.add_argument('--destination',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
 manifest=json.loads((ROOT/'manifests/upstreams.json').read_text());dest=a.destination.expanduser().absolute()
 print(json.dumps({'destination':str(dest),'sources':manifest,'execute':a.execute},indent=2))
 if not a.execute:return
 dest.mkdir(parents=True,exist_ok=True)
 for name,item in manifest.items():
  d=dest/name
  if not d.exists():run(['git','clone',item['url'],str(d)]);run(['git','-C',str(d),'checkout','--detach',item['commit']])
  head=subprocess.check_output(['git','-C',str(d),'rev-parse','HEAD'],text=True).strip()
  if head!=item['commit']:raise RuntimeError(f'{name}: wrong commit; refusing to reset existing checkout')
  for key in ('patch','installer_patch'):
   if item.get(key):patch(d,ROOT/item[key])
 expert=dest/'sage-expert'
 if not expert.exists():
  expert.mkdir()
  for name in ('IsaacLab','M2T2','server'):shutil.copytree(dest/'sage'/name,expert/name,symlinks=True)
  shutil.copytree(dest/'curobo',expert/'curobo',symlinks=True,ignore=shutil.ignore_patterns('.git','build','*.egg-info'))
 patch(expert,ROOT/'patches/expert-current.patch')
 print('Sources prepared. Set resources.expert_source; prepare M2T2 weight link and scene fixture separately. No environments were installed.')
if __name__=='__main__':main()
