import os,shutil,tempfile,unittest
from pathlib import Path
from verify_bundle import verify_bundle
class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=os.environ.get('EVOWORLD_PHASE7_BUNDLE')
        if not p:raise unittest.SkipTest('Set actual benchmark bundle')
        cls.bundle=Path(p)
    def test_native_stage_and_portable_hashes(self):self.assertTrue(verify_bundle(self.bundle)['passed'])
    def test_missing_photo_texture_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d)/'bundle';shutil.copytree(self.bundle,b);next((b/'assets/photo/textures').glob('*.png')).unlink()
            with self.assertRaises(ValueError):verify_bundle(b)
    def test_robot_layer_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            b=Path(d)/'bundle';shutil.copytree(self.bundle,b);(b/'robot/scene.usda').write_text('#usda 1.0\n')
            with self.assertRaises(ValueError):verify_bundle(b)
if __name__=='__main__':unittest.main()
