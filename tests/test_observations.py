import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from evoworld.observations import CONTRACT,apply_wrist,verify_artifact

class ObservationTests(unittest.TestCase):
 def test_actual_offset_and_metadata(self):
  camera=SimpleNamespace(offset=SimpleNamespace(pos=(0,0,0),rot=(1,0,0,0),convention='ros'))
  profile=apply_wrist(camera,{'observation_contract':CONTRACT})
  self.assertEqual(camera.offset.pos,(.07,0,.035))
  self.assertEqual(tuple(profile['rotation_ros_wxyz']),camera.offset.rot)
  self.assertEqual(profile['observation_contract'],CONTRACT)
 def test_reject_legacy_machine_contract(self):
  with self.assertRaises(ValueError):apply_wrist(None,{'observation_contract':'milk-bin-center-wrist-v1'})
 def test_reject_relabelled_old_artifact(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'source-manifest.json';p.write_text(json.dumps({'observation_contract':'milk-bin-center-wrist-v1'}))
   with self.assertRaises(ValueError):verify_artifact(Path(d),'train_data')
 def test_require_real_artifact_contract(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(FileNotFoundError):verify_artifact(Path(d),'checkpoint')
   (Path(d)/'observation-contract.json').write_text(json.dumps({'observation_contract':CONTRACT}))
   verify_artifact(Path(d),'checkpoint')
