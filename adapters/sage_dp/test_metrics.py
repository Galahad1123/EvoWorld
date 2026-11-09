import unittest
import numpy as np
from metrics import success_checks, initial_com_height

class MetricTests(unittest.TestCase):
    def test_initial_height_uses_com_before_first_recorded_action(self):
        rows=[{'attempt':0,'iteration':20,'object':[0,0,.825]+[0]*10},
              {'attempt':0,'iteration':21,'object':[0,0,.83]+[0]*10}]
        self.assertEqual(initial_com_height(rows,0),.825)
        with self.assertRaises(ValueError):initial_com_height(rows,1)
    def test_requires_lift_release_and_settling(self):
        s=np.zeros((60,13));s[:,2]=.8;s[10,2]=.85
        self.assertTrue(all(success_checks(s,.8,[0,0],[.04,.04]).values()))
        self.assertFalse(all(success_checks(s,.8,[0,0],[.001,.001]).values()))
        s[-1,7]=.1
        self.assertFalse(all(success_checks(s,.8,[0,0],[.04,.04]).values()))
        s[-1,7]=0;s[10,2]=.8
        self.assertFalse(all(success_checks(s,.8,[0,0],[.04,.04]).values()))
    def test_requires_goal_and_support(self):
        s=np.zeros((60,13));s[:,2]=.8;s[10,2]=.85
        self.assertFalse(all(success_checks(s,.8,[.2,0],[.04,.04]).values()))
        s[-1,2]=.7
        self.assertFalse(all(success_checks(s,.8,[0,0],[.04,.04]).values()))
if __name__=='__main__':unittest.main()
