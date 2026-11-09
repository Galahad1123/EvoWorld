import unittest
import numpy as np
from open_loop_metrics import action_errors, summarize, transition_mask, predict_recorded


class OpenLoopTests(unittest.TestCase):
    def actions(self, n=4):
        a=np.zeros((n,8));a[:,3]=1;a[:,7]=1
        return a

    def test_physical_units_and_quaternion_sign(self):
        ref=self.actions();pred=ref.copy()
        pred[:,0]=.03;pred[:,1]=.04;pred[:,3]=-2
        e=action_errors(pred,ref)
        np.testing.assert_allclose(e['position_cm'],5)
        np.testing.assert_allclose(e['rotation_deg'],0)
        pred[:,3:7]=[np.sqrt(.5),0,0,np.sqrt(.5)]
        np.testing.assert_allclose(action_errors(pred,ref)['rotation_deg'],90)

    def test_gripper_threshold_and_balanced_accuracy(self):
        ref=self.actions();ref[2:,7]=-1
        pred=ref.copy();pred[:,7]=[0,-.01,-.01,.01]
        e=action_errors(pred,ref);s=summarize(e)
        self.assertEqual(s['gripper_accuracy'],.5)
        self.assertEqual(s['gripper_balanced_accuracy'],.5)

    def test_reject_invalid_actions(self):
        a=self.actions()
        for bad in [a[:-1],np.full_like(a,np.nan),np.zeros_like(a)]:
            with self.assertRaises(ValueError):action_errors(bad,a)

    def test_switch_windows_clip_to_episode(self):
        a=self.actions(5);a[3:,7]=-1
        np.testing.assert_array_equal(transition_mask(a,1),[False,False,True,True,True])

    def test_alignment_queue_reset_and_no_padding(self):
        class QueuePolicy:
            def start_episode(self):self.queue=[]
            def __call__(self, obs):
                if not self.queue:self.queue=list(range(int(obs['t'][0]),int(obs['t'][0])+3))
                return np.full(8,self.queue.pop(0))
        obs={'t':np.arange(5)[:,None]*10}
        p=QueuePolicy()
        np.testing.assert_array_equal(predict_recorded(p,obs,'chunk8')[:,0],[0,1,2,30,31])
        np.testing.assert_array_equal(predict_recorded(p,obs,'fresh1')[:,0],[0,10,20,30,40])
        # A second episode must discard the first episode's unfinished chunk.
        np.testing.assert_array_equal(predict_recorded(p,{'t':np.array([[90]])},'chunk8')[:,0],[90])


if __name__=='__main__':unittest.main()
