import unittest
from dsrl_handoff import readiness

class HandoffTests(unittest.TestCase):
 def fixture(self):
  return dict(pipeline_exit=0,stage='baseline_ready_for_review',latest_step=20000,
   verified=True,checkpoint='/project/global_step_20000/actor',baseline_checkpoint='/project/global_step_20000/actor',
   baseline_complete=True,benchmark_valid=True,total=20,seeds=list(range(20000,20020)),successes=10,success_rate=.5,ready_for_dsrl=True)
 def test_valid(self):self.assertEqual(readiness(self.fixture()),'ready')
 def test_pending(self):self.assertEqual(readiness({'pipeline_exit':None}),'waiting')
 def test_bad_inputs(self):
  for key,value in [('pipeline_exit',1),('latest_step',18000),('verified',False),('baseline_complete',False),('benchmark_valid',False),('total',19),('seeds',list(range(20))),('success_rate',.8),('baseline_checkpoint','wrong'),('ready_for_dsrl',False),('successes',9)]:
   with self.subTest(key=key):
    data=self.fixture();data[key]=value
    with self.assertRaises(ValueError):readiness(data)
if __name__=='__main__':unittest.main()
