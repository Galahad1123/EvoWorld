"""Persist DSRL state omitted by the pinned native SAC checkpoint implementation."""
from functools import wraps
from pathlib import Path
import torch


def install_checkpoint_patch(policy):
 if getattr(policy,'_evoworld_resume_installed',False):return
 policy._evoworld_resume_installed=True
 native_save,native_load=policy.save_checkpoint,policy.load_checkpoint
 @wraps(native_save)
 def save(self,save_base_path,step):
  result=native_save(self,save_base_path,step)
  if not self.use_dsrl:return result
  payload={'version':1,'update_step':self.update_step,
           'target_shadow_f32':{n:v.detach().cpu() for n,v in self._target_shadow_f32.items()},
           'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
           'replay_rng':self.replay_buffer.random_generator.get_state()}
  path=Path(save_base_path)/f'evoworld-dsrl-state-rank-{self._rank}.pt'
  temporary=path.with_suffix('.pt.tmp');torch.save(payload,temporary);temporary.replace(path)
  return result
 @wraps(native_load)
 def load(self,load_base_path):
  path=Path(load_base_path)/f'evoworld-dsrl-state-rank-{self._rank}.pt'
  if self.use_dsrl and not path.exists():raise FileNotFoundError(f'Missing exact DSRL resume state: {path}')
  result=native_load(self,load_base_path)
  if not self.use_dsrl:return result
  payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
  assert payload['version']==1
  assert payload['target_shadow_f32'].keys()==self._target_shadow_f32.keys()
  for name,shadow in self._target_shadow_f32.items():shadow.copy_(payload['target_shadow_f32'][name])
  self.update_step=int(payload['update_step'])
  torch.set_rng_state(payload['torch_rng'])
  if payload['cuda_rng']:torch.cuda.set_rng_state_all(payload['cuda_rng'])
  self.replay_buffer.random_generator.set_state(payload['replay_rng'])
  return result
 policy.save_checkpoint=save;policy.load_checkpoint=load
