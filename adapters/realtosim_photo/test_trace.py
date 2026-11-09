import unittest
import numpy as np
from readback import validate_trace

class TraceTest(unittest.TestCase):
    def data(self):
        p=np.zeros((121,7));p[:,3]=1;p[:12,2]=np.linspace(.021,0,12)
        return np.arange(121)/60,p,np.zeros((121,3)),np.zeros((121,3))
    def test_real_motion_then_settling(self):self.assertGreater(validate_trace(*self.data())['drop_m'],.01)
    def test_reject_static(self):
        t,p,v,w=self.data();p[:,2]=.021
        with self.assertRaises(AssertionError):validate_trace(t,p,v,w)
    def test_reject_bad_time(self):
        t,p,v,w=self.data();t[61]=t[60]
        with self.assertRaises(AssertionError):validate_trace(t,p,v,w)
    def test_reject_tail_drift(self):
        t,p,v,w=self.data();p[-61:,0]=np.linspace(0,.02,61)
        with self.assertRaises(AssertionError):validate_trace(t,p,v,w)

if __name__=='__main__':unittest.main()
