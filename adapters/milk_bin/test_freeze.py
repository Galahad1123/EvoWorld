import unittest
import torch
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf_extension import freeze_sft_vlm
class FreezeTest(unittest.TestCase):
 def test_real_architecture_masks(self):
  with torch.device('meta'):
   model=Pi0(Pi0Config(pi05=True,action_horizon=20,discrete_state_input=True))
  model.llm.layers[0].pre_attention_norms[0].scale=torch.nn.Parameter(torch.full((2048,),.1427))
  norm_before=model.llm.layers[0].pre_attention_norms[0].scale.detach().clone()
  model.action_out_proj.weight=torch.nn.Parameter(torch.full((32,1024),.1234567))
  action_before=model.action_out_proj.weight.detach().clone()
  report=freeze_sft_vlm(model)
  self.assertTrue(torch.equal(norm_before,model.llm.layers[0].pre_attention_norms[0].scale))
  self.assertTrue(torch.equal(action_before,model.action_out_proj.weight))
  self.assertEqual(model.llm.layers[0].attn.q_proj[0].weight.dtype,torch.bfloat16)
  self.assertEqual(model.llm.layers[0].attn.q_proj[1].weight.dtype,torch.float32)
  self.assertEqual(model.img.stem.weight.dtype,torch.float32)
  self.assertGreater(report['frozen_parameters'],2_000_000_000)
  self.assertGreater(report['trainable_parameters'],250_000_000)
  self.assertLess(report['trainable_parameters'],500_000_000)
  self.assertTrue(all(not p.requires_grad for p in model.img.parameters()))
  self.assertTrue(all(not p.requires_grad for p in model.llm.embedder.parameters()))
  for block in model.llm.layers:
   self.assertTrue(all(not p.requires_grad for p in block.mlps[0].parameters()))
   self.assertTrue(all(p.requires_grad for p in block.mlps[1].parameters()))
  for name in ['action_in_proj','action_out_proj','time_mlp_in','time_mlp_out']:
   self.assertTrue(all(p.requires_grad for p in getattr(model,name).parameters()))
if __name__=='__main__':unittest.main()
