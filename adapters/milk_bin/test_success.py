import unittest
import numpy as np
from task_success import check_window
class SuccessTests(unittest.TestCase):
 def states(self):
  roots=np.zeros((51,13));roots[:,3]=1
  fingers=np.full((51,2),.04)
  vertices=np.array([[x,y,z] for x in [-.0275,.0275] for y in [-.019,.019] for z in [0,.13]])
  return roots,fingers,vertices
 def check(self,r,f,v,c=None):return check_window(r,f,v,np.zeros(3),[.0876,.0926],0,.02,gripper_commands=np.ones(len(r)) if c is None else c)['success']
 def test_above_rim_allowed(self):self.assertTrue(self.check(*self.states()))
 def test_outside(self):
  r,f,v=self.states();r[:,0]=.03;self.assertFalse(self.check(r,f,v))
 def test_hover(self):
  r,f,v=self.states();r[:,2]=.015;self.assertFalse(self.check(r,f,v))
 def test_still_gripped(self):
  r,f,v=self.states();f[:]=.015;self.assertFalse(self.check(r,f,v))
 def test_transient(self):
  r,f,v=self.states();r[0,0]=.1;self.assertFalse(self.check(r,f,v))
 def test_too_short(self):
  r,f,v=self.states();self.assertFalse(self.check(r[:25],f[:25],v))
 def test_nan(self):
  r,f,v=self.states();r[-1,0]=np.nan;self.assertFalse(self.check(r,f,v))
 def test_broad_face_grasp(self):
  r,f,v=self.states();f[:]=.0275;self.assertFalse(self.check(r,f,v))
 def test_close_command_even_if_fingers_open(self):
  r,f,v=self.states();self.assertFalse(self.check(r,f,v,-np.ones(len(r))))
 def test_close_command_earlier_in_window(self):
  r,f,v=self.states();c=np.ones(len(r));c[0]=-1;self.assertFalse(self.check(r,f,v,c))
 def test_partially_open_is_not_released(self):
  r,f,v=self.states();f[:]=.036;self.assertFalse(self.check(r,f,v))
 def test_motion(self):
  r,f,v=self.states();r[-1,7]=.1;self.assertFalse(self.check(r,f,v))
if __name__=='__main__':unittest.main()
