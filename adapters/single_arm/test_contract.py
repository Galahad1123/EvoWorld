import unittest
import numpy as np
from contract import RobotAdapter, TaskAdapter, pose_matrix, transform_pose, scene_from_realtosim, scene_from_sage

class ContractTests(unittest.TestCase):
    def setUp(self): self.a=RobotAdapter()
    def test_unbound_live_interface_fails_closed(self):
        with self.assertRaises(RuntimeError):self.a.reset()
        with self.assertRaises(RuntimeError):self.a.step([.4,0,.5,1,0,0,0,1])
        with self.assertRaises(RuntimeError):TaskAdapter().reset()
    def test_rotated_base_roundtrip(self):
        # 90deg yaw and translation must rotate axes as well as translate.
        base=[2,3,1,2**-.5,0,0,2**-.5]; p=[1,0,.2,1,0,0,0]
        w=transform_pose(p,pose_matrix(base));np.testing.assert_allclose(w[:3],[2,4,1.2],atol=1e-9)
        np.testing.assert_allclose(pose_matrix(transform_pose(w,np.linalg.inv(pose_matrix(base)))),pose_matrix(p),atol=1e-9)
    def test_units_quaternion_and_nonrigid_rejected(self):
        for p in ([0,0,0,0,0,0,0],[0,0,0,2,0,0,0],[0,0,np.nan,1,0,0,0]):
            with self.assertRaises(ValueError):pose_matrix(p)
        with self.assertRaises(ValueError):transform_pose([0,0,0,1,0,0,0],np.diag([2,1,1,1]))
        with self.assertRaises(ValueError):self.a.action([0,0,.4,1,0,0,0,1],units='mm')
        with self.assertRaises(ValueError):self.a.action([0,0,.4,1,0,0,0,1],frame='world')
    def test_action_bounds_rate_and_sign(self):
        good=[.4,0,.5,1,0,0,0,1];np.testing.assert_equal(self.a.action(good),good)
        for a in ([4,0,.5,1,0,0,0,1],[.4,0,.5,1,0,0,0,0],[.4,0,.5,1,0,0,0,np.nan],[0]*29):
            with self.assertRaises(ValueError):self.a.action(a)
        with self.assertRaises(ValueError):self.a.action([.5,0,.5,1,0,0,0,1],previous=good)
        with self.assertRaises(ValueError):self.a.action([.4,0,.5,0,1,0,0,1],previous=good)
    def test_joint_names_fail_closed(self):
        self.a.check_joints(self.a.joint_names)
        with self.assertRaises(ValueError):self.a.check_joints(self.a.joint_names[::-1])
    def test_reset_snapshot_is_independent(self):
        state={'root':np.array([1.,2.]),'q':np.zeros(9),'nested':{'action':np.ones(8)}}
        snap=self.a.snapshot(state);state['nested']['action'][0]=9
        restored=self.a.restore(snap);restored['q'][0]=5
        self.assertEqual(snap['q'][0],0);self.assertEqual(snap['nested']['action'][0],1)
    def test_scene_sources_and_orientation(self):
        obj={'id':'milk','size_m':[.055,.038,.13],'position_m':[0,0,.001],'quaternion_wxyz':[1,0,0,0],'asset':'assets/milk.usda','evidence':{'scale':'measured'},'physics':{'mass_kg':.235}}
        s=scene_from_realtosim({'units':'m','up_axis':'Z','objects':[obj]},np.eye(4));self.assertEqual(s['objects'][0]['evidence'],obj['evidence']);self.assertEqual(s['objects'][0]['asset'],'assets/milk.usda')
        t={'a':{'position':{'x':1,'y':2,'z':3},'rotation':{'x':0,'y':0,'z':90}}}
        s=scene_from_sage(t,{'a':{'static':False,'mass':.08}},np.eye(4));np.testing.assert_allclose(pose_matrix(s['objects'][0]['pose_world'])[:3,:3]@[1,0,0],[0,1,0],atol=1e-9)
        with self.assertRaises(ValueError):scene_from_realtosim({'units':'m','up_axis':'Z','objects':[obj,obj]},np.eye(4))
    def test_task_failure_labels_and_measured_contact(self):
        t=TaskAdapter();self.assertFalse(t.contact_success(.03,0,0,.04))
        self.assertFalse(t.contact_success(.001,.001,10,.04))
        self.assertTrue(t.contact_success(.025,.001,4,.04))
        self.assertEqual(t.failure_label('reset'), 'reset_mismatch')
        with self.assertRaises(ValueError):t.failure_label('invented')
    def test_missing_scene_units_fail_closed(self):
        obj={'id':'a','asset':'a.usda','size_m':[1,1,1],'position_m':[0,0,0],'quaternion_wxyz':[1,0,0,0],'physics':{},'evidence':{}}
        for spec in ({'objects':[obj]},{'units':'m','objects':[obj]}):
            with self.assertRaises(ValueError):scene_from_realtosim(spec,np.eye(4))
    def test_reject_g1_and_photo_state_contract(self):
        for schema in ('g1','evoworld.photo_asset_physics.v1'):
            with self.assertRaises(ValueError):self.a.check_schema(schema)

if __name__=='__main__':unittest.main()
