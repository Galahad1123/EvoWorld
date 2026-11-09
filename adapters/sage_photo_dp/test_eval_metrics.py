import unittest
import numpy as np
from eval_metrics import physical_result,validated_action
class EvaluationTests(unittest.TestCase):
 def fixture(self):
  s=np.zeros((50,13));s[:,3]=1;s[:,2]=.85;s[10:20,2]=.95
  r=s.copy();r[:,2]-=.05
  v=np.array([[x,y,z] for x in [-.02,.02] for y in [-.02,.02] for z in [0,.1]])
  return s,r,v
 def test_valid_upright_release(self):
  s,r,v=self.fixture();a=physical_result(s,r,.85,[0,0],v,.8,[.04,.04])
  self.assertTrue(a['success'])
 def test_spin_and_wrong_support_rejected(self):
  s,r,v=self.fixture();s[-25:,12]=.3
  self.assertFalse(physical_result(s,r,.85,[0,0],v,.8,[.04,.04])['checks']['angular_stable'])
  s[-25:,12]=0;r[-1,2]+=.02
  self.assertFalse(physical_result(s,r,.85,[0,0],v,.8,[.04,.04])['checks']['support'])
 def test_gripper_and_lift_required(self):
  s,r,v=self.fixture();s[:,2]=.85
  result=physical_result(s,r,.85,[0,0],v,.8,[0,0])
  self.assertFalse(result['success']);self.assertFalse(result['checks']['lift']);self.assertFalse(result['checks']['released'])
 def test_invalid_actions_and_runtime_projection(self):
  for bad in [np.zeros(7),np.zeros(8),np.full(8,np.nan)]:
   with self.assertRaises(ValueError):validated_action(bad)
  good=validated_action([.5,0,.2,2,0,0,0,0])
  np.testing.assert_allclose(good,[.5,0,.2,1,0,0,0,1])
if __name__=='__main__':unittest.main()
