import unittest
from pathlib import Path
import torch
from hydra import initialize_config_dir,compose
from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler
class ScheduleTests(unittest.TestCase):
 def test_native_group_lr_reaches_intended_floor(self):
  with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent),version_base='1.1'):c=compose(config_name='sft_d46')
  o=c.actor.optim
  optimizer=torch.optim.AdamW([{'params':[torch.nn.Parameter(torch.zeros(1))],'lr':o.lr}])
  scheduler=get_lr_scheduler(o.lr_scheduler,optimizer,o.lr_warmup_steps,o.total_training_steps,min_lr=o.min_lr,min_lr_rate=o.get('min_lr_rate'))
  actual=lambda step:scheduler.base_lrs[0]*scheduler.lr_lambdas[0](step)
  self.assertAlmostEqual(actual(1000),1e-4)
  self.assertAlmostEqual(actual(20000),1e-5)
if __name__=='__main__':unittest.main()
