import unittest
import numpy as np
from dwell import prepend_dwell
class DwellTest(unittest.TestCase):
 def test_fifteen_control_ticks(self):
  current=np.array([1.,2,3,1,0,0,0]);path=np.ones((20,7))
  out=prepend_dwell(path,current,.02,.3)
  self.assertEqual(len(out),35)
  np.testing.assert_array_equal(out[:15],np.tile(current,(15,1)))
  np.testing.assert_array_equal(out[15:],path)
 def test_round_up(self):
  self.assertEqual(len(prepend_dwell(np.ones((2,7)),np.zeros(7),.08,.3)),6)
if __name__=='__main__':unittest.main()
