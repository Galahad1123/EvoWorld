import copy,json,unittest
from pathlib import Path
import numpy as np
from specs import make_specs,validate_specs

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/phase5-photo-f92ngyB1/round3/build'
class SpecTests(unittest.TestCase):
    def setUp(self):
        self.source=json.loads((SOURCE/'scene.json').read_text());self.meta=json.loads((SOURCE/'metadata.json').read_text())
        self.scene,self.robot,self.task=make_specs(self.source,self.meta)
    def test_known_world_placement_and_measured_anchor(self):
        self.assertTrue(validate_specs(self.scene,self.robot,self.task))
        self.assertEqual({o['id'] for o in self.scene['objects']},{'milk','tissue','desk','tray','divider','monitor'})
        milk=self.scene['objects'][0];np.testing.assert_allclose(milk['pose_world'][:3],[0,0,.751]);np.testing.assert_allclose(self.robot['base_pose_world'][:3],[0,-.48,.751]);self.assertEqual(milk['evidence']['geometry']['measured_reference_envelope_m'],[.055,.038,.130])
        np.testing.assert_allclose(self.robot['base_pose_world'][3:], [2**-.5,0,0,2**-.5],atol=1e-12)
    def test_source_geometry_and_physics_preserved(self):
        for o,original in zip(self.scene['objects'],self.source['objects']):
            self.assertEqual(o['source_record'],original);self.assertEqual(o['physics'],original['physics']);self.assertEqual(o['size_m'],original['size_m'])
        self.assertEqual([o['id'] for o in self.scene['objects'] if o['dynamic']],['milk','tissue'])
    def test_reject_bad_units_and_scaled_frame(self):
        for field,value in [('units','cm'),('up_axis','Y')]:
            s=copy.deepcopy(self.scene);s[field]=value
            with self.assertRaises(ValueError):validate_specs(s,self.robot,self.task)
        s=copy.deepcopy(self.scene);s['world_T_source'][0][0]=100
        with self.assertRaises(ValueError):validate_specs(s,self.robot,self.task)
    def test_reject_duplicates_missing_or_changed_pose(self):
        for change in [lambda s:s['objects'].append(s['objects'][0]),lambda s:s['objects'].pop(),lambda s:s['objects'][0]['pose_world'].__setitem__(2,1.501)]:
            s=copy.deepcopy(self.scene);change(s)
            with self.assertRaises(ValueError):validate_specs(s,self.robot,self.task)
    def test_reject_invalid_robot_and_unattributed_placement(self):
        r=copy.deepcopy(self.robot);r['base_pose_world'][3:]=[0,0,0,0]
        with self.assertRaises(ValueError):validate_specs(self.scene,r,self.task)
        r=copy.deepcopy(self.robot);del r['provenance']['base_pose_world']
        with self.assertRaises(ValueError):validate_specs(self.scene,r,self.task)
    def test_reject_ignored_fixed_profile_fields(self):
        for target,key,value in [('task','gripper',-1),('robot','prim_path','/missing'),('robot','tool_offset_m',[0,0,.1]),('scene','generation_execution','new_inference'),('object','prim_path','/missing')]:
            with self.subTest(target=target,key=key):
                scene,robot,task=copy.deepcopy((self.scene,self.robot,self.task))
                item={'scene':scene,'robot':robot,'task':task,'object':scene['objects'][0]}[target];item[key]=value
                with self.assertRaises(ValueError):validate_specs(scene,robot,task)
    def test_reject_escape_paths_and_fake_mode(self):
        for path in ['/tmp/a.usda','../a.usda']:
            s=copy.deepcopy(self.scene);s['objects'][0]['asset']=path
            with self.assertRaises(ValueError):validate_specs(s,self.robot,self.task)
        s=copy.deepcopy(self.scene);s['mode']='new_gpt_generation'
        with self.assertRaises(ValueError):validate_specs(s,self.robot,self.task)

if __name__=='__main__':unittest.main()
