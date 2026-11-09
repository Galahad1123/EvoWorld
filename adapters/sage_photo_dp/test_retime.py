import unittest
import numpy as np
from retime import retime_pose
class RetimingTests(unittest.TestCase):
 def test_slowdown_and_endpoints(self):
  p=np.zeros((9,7));p[:,0]=np.arange(9)*.01;p[:,3]=1
  q=retime_pose(p,.8)
  self.assertEqual(len(q),11)
  np.testing.assert_allclose(q[[0,-1]],p[[0,-1]])
  np.testing.assert_allclose(np.diff(q[:,0]),.008)
 def test_sign_equivalent_rotation(self):
  p=np.zeros((3,7));p[:,3]=[1,-1,1]
  q=retime_pose(p,.8)
  np.testing.assert_allclose(np.abs(q[:,3]),1)
 def test_invalid(self):
  with self.assertRaises(ValueError):retime_pose(np.zeros((3,7)),.8)
if __name__=='__main__':unittest.main()
