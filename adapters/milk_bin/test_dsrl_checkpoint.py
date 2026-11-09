import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
from dsrl_checkpoint import install_checkpoint_patch

class TinyStrategy:
 def save_checkpoint(self,model,save_path,**kwargs):
  path=Path(save_path);path.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),path/'model.pt')
 def load_checkpoint(self,model,load_path,**kwargs):model.load_state_dict(torch.load(Path(load_path)/'model.pt'))
 def get_model_state_dict(self,model,**kwargs):return model.state_dict()
 def load_model_with_state_dict(self,model,state,**kwargs):model.load_state_dict(state)

class TinyReplay:
 def __init__(self):self.random_generator=torch.Generator().manual_seed(42)
 def save_checkpoint(self,path):Path(path).mkdir(parents=True,exist_ok=True)
 def load_checkpoint(self,path):self.random_generator.manual_seed(42)

class CheckpointTests(unittest.TestCase):
 def worker(self):
  return NS(model=torch.nn.Linear(2,1),target_model=torch.nn.Linear(2,1),use_dsrl=True,
   cfg=NS(actor=NS(fsdp_config=NS(use_orig_params=True))),is_weight_offloaded=False,is_optimizer_offloaded=False,
   _strategy=TinyStrategy(),optimizer=None,qf_optimizer=None,lr_scheduler=None,qf_lr_scheduler=None,alpha_optimizer=None,
   _rank=0,replay_buffer=TinyReplay(),update_step=0,_target_shadow_f32={'q_head.weight':torch.zeros(1,2)})
 def test_native_checkpoint_resumes_update_counter_shadow_and_replay_rng(self):
  install_checkpoint_patch(EmbodiedSACFSDPPolicy)
  original=self.worker();original.update_step=7
  original._target_shadow_f32['q_head.weight'].fill_(.1234567)
  torch.rand(10,generator=original.replay_buffer.random_generator)
  with tempfile.TemporaryDirectory() as temp:
   EmbodiedSACFSDPPolicy.save_checkpoint(original,temp,3)
   restored=self.worker()
   EmbodiedSACFSDPPolicy.load_checkpoint(restored,temp)
   self.assertEqual(restored.update_step,7)
   self.assertTrue(torch.equal(restored._target_shadow_f32['q_head.weight'],original._target_shadow_f32['q_head.weight']))
   self.assertTrue(torch.equal(restored.replay_buffer.random_generator.get_state(),original.replay_buffer.random_generator.get_state()))

if __name__=='__main__':unittest.main()
