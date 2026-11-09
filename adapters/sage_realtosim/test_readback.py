"""Acceptance regressions run against actual recorded integration evidence."""
import json,os,shutil,tempfile,unittest
from pathlib import Path
from readback import verify
class ReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=os.environ.get('EVOWORLD_PHASE7_RUN')
        if not p:raise unittest.SkipTest('Set EVOWORLD_PHASE7_RUN to real simulation output')
        cls.run_dir=Path(p)
    def test_recorded_integration(self):self.assertTrue(verify(self.run_dir)['passed'])
    def corrupt(self,name,mutate):
        with tempfile.TemporaryDirectory() as d:
            for p in self.run_dir.glob('*'):
                if p.is_file() and p.suffix in ('.json','.npz'):shutil.copy2(p,Path(d)/p.name)
            p=Path(d)/name;x=json.loads(p.read_text());mutate(x);p.write_text(json.dumps(x))
            with self.assertRaises((ValueError,AssertionError)):verify(d)
    def test_no_motion_rejected(self):self.corrupt('trace.json',lambda rows:[r.update(hand_base=rows[99]['hand_base']) for r in rows])
    def test_wrong_base_rejected(self):self.corrupt('reset.json',lambda r:r['root'].__setitem__(1,0.48))
    def test_object_falling_rejected(self):self.corrupt('trace.json',lambda rows:[r['objects']['milk'].__setitem__(2,0.2) for r in rows[-10:]])
    def test_frozen_dynamic_object_rejected(self):self.corrupt('trace.json',lambda rows:[r['objects'].update(milk=[0,0,.751,1,0,0,0,0,0,0,0,0,0]) for r in rows])
    def test_unintended_contact_rejected(self):self.corrupt('trace.json',lambda rows:rows[-1]['contact_forces_world'][1].__setitem__(0,50))
    def test_incomplete_rejected(self):self.corrupt('trace.json',lambda rows:rows.pop())
    def test_stale_wrist_rejected(self):self.corrupt('trace.json',lambda rows:rows[30]['wrist'].update(position_world=[0,0,0]))
if __name__=='__main__':unittest.main()
