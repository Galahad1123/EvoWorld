import unittest
from train_photo import build_config
class TrainingTests(unittest.TestCase):
 def test_budget_and_inputs(self):
  c=build_config('/tmp/unused')
  self.assertEqual(c.train.batch_size,32)
  self.assertEqual(c.train.num_epochs*c.experiment.epoch_every_n_steps,20000)
  self.assertEqual(c.algo.optim_params.policy.learning_rate.warmup_steps,1000)
  self.assertEqual(c.experiment.save.every_n_epochs*c.experiment.epoch_every_n_steps,2000)
  self.assertEqual(list(c.observation.modalities.obs.rgb),['rgb_tabletop','rgb_wrist'])
if __name__=='__main__':unittest.main()
