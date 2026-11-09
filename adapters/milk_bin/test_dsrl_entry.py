"""Regression: Ray snapshots methods before worker-process extension registration."""
import subprocess
import sys
import unittest

class EntryTests(unittest.TestCase):
 def test_actor_hooks_are_registered_before_ray_copies_methods(self):
  code = '''
import dsrl_entry
import ray
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy as Policy
actual = ray.remote(Policy).__ray_metadata__.modified_class
for name in ('init_worker', 'run_training', 'save_checkpoint', 'load_checkpoint'):
 assert 'audited_' in getattr(actual,name).__qualname__, (name,getattr(actual,name))
assert 'setup_with_fp32_alpha' in actual.setup_model_and_optimizer.__qualname__
'''
  subprocess.run([sys.executable,'-c',code],check=True)

if __name__=='__main__': unittest.main()
