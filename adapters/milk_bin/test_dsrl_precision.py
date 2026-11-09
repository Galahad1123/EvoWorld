import unittest
from types import SimpleNamespace
import torch
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from dsrl_precision import prepare_entropy_precision

class PrecisionTests(unittest.TestCase):
 def test_alpha_updates_at_native_lr_without_replacing_optimizer_parameter(self):
  temperature=EntropyTemperature(initial_alpha=1.,dtype=torch.bfloat16)
  optimizer=torch.optim.Adam(temperature.parameters(),lr=3e-4)
  worker=SimpleNamespace(entropy_temp=temperature,alpha_optimizer=optimizer)
  original_parameter=temperature.base_alpha
  prepare_entropy_precision(worker)
  before=temperature.base_alpha.detach().clone()
  for _ in range(3):
   optimizer.zero_grad();(temperature.compute_alpha()*10).backward();optimizer.step()
  self.assertFalse(torch.equal(before,temperature.base_alpha),'BF16 rounded away all alpha updates')
  self.assertIs(temperature.base_alpha,original_parameter)
  self.assertIs(optimizer.param_groups[0]['params'][0],temperature.base_alpha)
  self.assertEqual(temperature.base_alpha.dtype,torch.float32)

if __name__=='__main__':unittest.main()
