"""在已有独立专家环境中编译扩展；默认只打印计划。"""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoworld.config import load
from evoworld.expert import compiler_environment,site_packages,prepare_cuda_link

def main():
 p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--uv',default='uv');p.add_argument('--execute',action='store_true');a=p.parse_args();c=load(a.config)
 source=Path(c['resources']['expert_source']);nv=Path(c['upstreams']['nvdiffrast']);py=c['environments']['expert']
 items=[source/'M2T2/pointnet2_ops',source/'M2T2',source/'curobo',nv]
 print('\n'.join(str(x) for x in items))
 if not a.execute:return
 env=compiler_environment(c);env['UV_CACHE_DIR']=str(Path(c['paths']['cache'])/'uv')
 site=site_packages(c);prepare_cuda_link(site)
 experience=Path(c['isaac_experience'])
 (site/'isaacsim_base.pth').write_text(str(experience.parents[2])+'\n')
 for item in items:
  if not item.exists():raise FileNotFoundError(item)
  build_env=dict(env)
  if item==source/'curobo':build_env['SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO']='0.7.8'
  subprocess.run([a.uv,'pip','install','--python',py,'--no-deps','--no-build-isolation',str(item)],env=build_env,check=True)
if __name__=='__main__':main()
