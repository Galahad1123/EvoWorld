"""Check independent RLinf runtime without loading pretrained weights."""
import json,sys,os,importlib,importlib.metadata as md
from pathlib import Path
import torch
import ray
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
 out=ROOT/'logs/rlinf-setup';out.mkdir(parents=True,exist_ok=True)
 assert Path(sys.prefix).resolve()==(ROOT/'envs/rlinf-py312').resolve()
 assert not any('envs/sage' in p or 'envs/isaacsim' in p for p in sys.path)
 versions={n:md.version(n) for n in ['torch','torchvision','ray','rlinf-openpi','numpy','isaaclab']}
 modules=['rlinf.scheduler','rlinf.algorithms','rlinf.models.embodiment.openpi.pi0','rlinf.models.embodiment.openpi.pi0_config','rlinf.envs.sim.isaaclab', 'rlinf.models.embodiment.openpi.tasks.rl', 'rlinf.models.embodiment.openpi.dataconfig', 'rlinf.models.embodiment.openpi.policies.franka_policy', 'openpi.transforms']
 for name in modules:importlib.import_module(name)
 assert torch.cuda.is_available()
 x=torch.randn(64,64,device='cuda',requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
 assert torch.isfinite(x.grad).all()
 with initialize_config_dir(config_dir=str(ROOT/'upstream/RLinf/examples/embodiment/config'),version_base='1.1'):
  cfg=compose(config_name='isaaclab_franka_stack_cube_ppo_openpi_pi05',overrides=['env.train.total_num_envs=1','env.eval.total_num_envs=1'])
  OmegaConf.save(cfg,out/'official-config-composed.yaml')
 assert cfg.actor.model.model_type=='openpi'
 ray.init(address='local',num_cpus=2,num_gpus=1,include_dashboard=False,object_store_memory=256*1024*1024,_temp_dir='/tmp/evoworld-rlinf-ray')
 try:
  @ray.remote
  def check():
   import rlinf,torch
   return {'torch':torch.__version__,'rlinf':rlinf.__file__}
  worker=ray.get(check.remote(),timeout=60)
 finally:ray.shutdown()
 result={'passed':True,'versions':versions,'python':sys.version,'prefix':sys.prefix,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'capability':torch.cuda.get_device_capability(),'cuda_backward_finite':True,'ray_worker':worker,'model_weights_loaded':False,'imported_modules':modules}
 (out/'runtime-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
