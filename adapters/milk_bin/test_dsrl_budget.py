import tempfile,unittest
from pathlib import Path
from dsrl_budget import install_budget,retain_completed

class BudgetTests(unittest.TestCase):
 def test_budget_saves_then_finishes(self):
  class Runner:
   def run(self):self._log_step_metrics();self.extra_update=True
   def _log_step_metrics(self):self.logged=True
   def _save_checkpoint(self):self.saved=True
   def _finish_run(self):self.finished=True
  install_budget(Runner,deadline=1000,clock=lambda:800,retention=None)
  r=Runner();r.run()
  self.assertTrue(r.saved and r.finished and r.logged);self.assertFalse(hasattr(r,'extra_update'))
 def test_retention_requires_completion_and_preserves_unrelated(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for i in [1,2,3]:
    p=root/f'global_step_{i}';p.mkdir();(p/'completed.json').write_text('{}')
   (root/'global_step_4').mkdir();(root/'notes').mkdir()
   retain_completed(root,2)
   self.assertFalse((root/'global_step_1').exists())
   self.assertTrue(all((root/n).exists() for n in ['global_step_2','global_step_3','global_step_4','notes']))
if __name__=='__main__':unittest.main()
