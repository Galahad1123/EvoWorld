import unittest
import torch
from dsrl_evidence import parameter_report, assert_progress, worker_report
from types import SimpleNamespace as NS

class EvidenceTests(unittest.TestCase):
 def model(self):
  m=torch.nn.Module()
  m.base=torch.nn.Linear(2,2);m.base.requires_grad_(False)
  m.dsrl_action_noise_net=torch.nn.Linear(2,2)
  m.q_head=torch.nn.Linear(2,1)
  return m
 def test_detects_accidentally_unfrozen_base(self):
  m=self.model();m.base.weight.requires_grad_(True)
  with self.assertRaisesRegex(AssertionError,'Unexpected trainable'):parameter_report(m)
 def test_detects_frozen_weight_drift(self):
  m=self.model();before=parameter_report(m)
  with torch.no_grad():
   m.base.weight.add_(1);m.q_head.weight.add_(1);m.dsrl_action_noise_net.weight.add_(1)
  with self.assertRaisesRegex(AssertionError,'Frozen decoder'):assert_progress(before,parameter_report(m))
 def test_requires_both_actor_and_critic_update(self):
  m=self.model();before=parameter_report(m)
  with torch.no_grad():m.q_head.weight.add_(1)
  with self.assertRaisesRegex(AssertionError,'Noise actor'):assert_progress(before,parameter_report(m))
  with torch.no_grad():m.dsrl_action_noise_net.weight.add_(1)
  assert_progress(before,parameter_report(m))

class ResumeEvidenceTests(unittest.TestCase):
 def test_restore_evidence_detects_same_size_replay_corruption_and_rng_drift(self):
  model=EvidenceTests().model();optim=torch.optim.Adam(model.parameters())
  alpha=torch.nn.Linear(1,1);alpha_optim=torch.optim.Adam(alpha.parameters())
  replay_data={'actions':torch.ones(2,20,32)}
  class Replay:
   def __init__(self):
    self.random_generator=torch.Generator().manual_seed(1)
    self._flat_trajectory_cache=NS(cache={0:0},get=lambda i:replay_data)
   def __len__(self):return 2
  scheduler=torch.optim.lr_scheduler.ConstantLR(optim,factor=1.)
  worker=NS(model=model,target_model=model,optimizer=optim,qf_optimizer=optim,alpha_optimizer=alpha_optim,
   entropy_temp=alpha,_target_shadow_f32={'base.weight':torch.ones(2,2)},update_step=2,replay_buffer=Replay(),
   lr_scheduler=scheduler,qf_lr_scheduler=scheduler,alpha_lr_scheduler=scheduler)
  before=worker_report(worker)
  required={'replay_contents','replay_rng','torch_rng','cuda_rng','schedulers'}
  self.assertTrue(required.issubset(before),'Resume audit omits replay/RNG/scheduler state')
  replay_data['actions'][0,0,0]=2
  after=worker_report(worker)
  self.assertEqual(before['replay_size'],after['replay_size'])
  self.assertNotEqual(before['replay_contents'],after['replay_contents'])
  torch.rand(1,generator=worker.replay_buffer.random_generator)
  self.assertNotEqual(after['replay_rng'],worker_report(worker)['replay_rng'])
  worker._target_shadow_f32['base.weight'].add_(1)
  self.assertNotEqual(after['target_shadow'],worker_report(worker)['target_shadow'])

if __name__=='__main__':unittest.main()
