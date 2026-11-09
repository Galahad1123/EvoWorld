"""Regression negatives mutate real recorded traces, not synthetic passing fixtures."""
import json,os,shutil,tempfile,unittest
from pathlib import Path
from readback import verify

class ReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('EVOWORLD_PHASE6_ACCEPTED')
        if not path:raise unittest.SkipTest('Set EVOWORLD_PHASE6_ACCEPTED to actual audit run')
        cls.source=Path(path)
    def test_real_trace(self):self.assertTrue(verify(self.source)['passed'])
    def bad(self,file,change):
        with tempfile.TemporaryDirectory() as d:
            for p in self.source.iterdir():
                if p.is_file() and p.suffix in ('.json','.npz','.txt'):shutil.copy2(p,Path(d)/p.name)
            p=Path(d)/file;v=json.loads(p.read_text());change(v);p.write_text(json.dumps(v))
            with self.assertRaises((AssertionError,ValueError)):verify(d)
    def test_stale_wrist_rejected(self):self.bad('runtime.json',lambda v:v['cameras']['wrist'].update(position_world=[.088,0,.846]))
    def test_camera_rotation_rejected(self):self.bad('runtime.json',lambda v:v['cameras']['wrist'].update(quaternion_ros_wxyz=[1,0,0,0]))
    def test_intermediate_wrist_pose_rejected(self):self.bad('trace.json',lambda v:v[10].update(wrist_position_world=[0,0,0]))
    def test_camera_clock_rejected(self):self.bad('trace.json',lambda v:v[10]['camera_time_s'].update(wrist=0))
    def test_false_contact_rejected(self):self.bad('trace.json',lambda v:[r.update(finger_force_world=[[0,0,0],[0,0,0]]) for r in v if r['label']=='contact_close'])
    def test_nonrepeatable_reset_rejected(self):self.bad('runtime.json',lambda v:v['reset_states'][1]['q'][0].__setitem__(0,.2))
    def test_joint_reordering_rejected(self):self.bad('runtime.json',lambda v:v['joint_names'].reverse())
    def test_incomplete_trace_rejected(self):self.bad('trace.json',lambda v:v.pop())

if __name__=='__main__':unittest.main()
