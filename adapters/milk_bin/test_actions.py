import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from actions import absolute_to_delta,delta_to_absolute
class ActionsTest(unittest.TestCase):
 def test_roundtrip_nonidentity_base_pose(self):
  rng=np.random.default_rng(20)
  poses=np.c_[rng.uniform(-.4,.4,(100,3)),Rotation.random(100,random_state=rng).as_quat(scalar_first=True)]
  goals=np.c_[rng.uniform(-.4,.4,(100,3)),Rotation.random(100,random_state=rng).as_quat(scalar_first=True),rng.choice([-1.,1.],100)]
  result=delta_to_absolute(poses,absolute_to_delta(poses,goals))
  np.testing.assert_allclose(result[:,:3],goals[:,:3],atol=1e-6)
  angle=(Rotation.from_quat(result[:,3:7],scalar_first=True)*Rotation.from_quat(goals[:,3:7],scalar_first=True).inv()).magnitude()
  np.testing.assert_allclose(angle,0,atol=1e-6)
  np.testing.assert_array_equal(result[:,-1],goals[:,-1])
 def test_gripper_sign(self):
  pose=np.array([[0,0,0,1,0,0,0.]])
  out=delta_to_absolute(pose,np.array([[0,0,0,0,0,0,-.1]]));self.assertEqual(out[0,-1],-1)
if __name__=='__main__':unittest.main()
