"""专家环境的进程级配置；不改变其他环境或系统 CUDA。"""
import os,subprocess
from pathlib import Path

def site_packages(c):
 if c.get('expert_site_packages'):return Path(c['expert_site_packages'])
 py=c['environments']['expert']
 return Path(subprocess.check_output([py,'-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'],text=True).strip())

def compiler_environment(c,environ=None):
 e=dict(os.environ if environ is None else environ);site=site_packages(c);cuda=site/'nvidia/cu13'
 e['CUDA_HOME']=str(cuda)
 bins=[str(cuda/'bin')]
 if c.get('environments',{}).get('expert'):bins.append(str(Path(c['environments']['expert']).parent))
 e['PATH']=os.pathsep.join(bins+[e.get('PATH','')])
 includes=[str(p) for p in (site/'nvidia').glob('*/include') if p.is_dir()]
 cccl=cuda/'include/cccl'
 if cccl.is_dir():includes.append(str(cccl))
 e['CPATH']=os.pathsep.join(includes+([e['CPATH']] if e.get('CPATH') else []))
 e['TORCH_CUDA_ARCH_LIST']=str(c.get('expert_cuda_arch','12.0'))
 e['MAX_JOBS']=str(c.get('expert_build_jobs',4));e['CMAKE_BUILD_PARALLEL_LEVEL']=e['MAX_JOBS'];e['PYTHONNOUSERSITE']='1'
 return e

def prepare_cuda_link(site):
 lib=Path(site)/'nvidia/cu13/lib';target=lib/'libcudart.so.13';link=lib/'libcudart.so'
 if link.exists():return
 if not target.is_file():raise FileNotFoundError(target)
 # Versioned runtime wheels omit the unversioned development linker name.
 # Never replace an existing broken link or touch system-wide CUDA paths.
 link.symlink_to(target.name)
