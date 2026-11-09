import tempfile, unittest, os
from pathlib import Path
from evoworld.config import load, credentials
class ConfigTests(unittest.TestCase):
 def test_relocation(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'local.yaml';p.write_text('storage_root: data\n')
   c=load(p);self.assertEqual(c['paths']['models'],str(Path(d)/'data/models'))
 def test_env_priority_and_no_shell(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'key.env';p.write_text('OPENAI_API_KEY=file-key\nBAD=$(touch /tmp/should-not-exist)\n')
   self.assertEqual(credentials(p,{'OPENAI_API_KEY':'env-key'}),{'OPENAI_API_KEY':'env-key'})
 def test_relative_override(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'local.yaml';p.write_text('storage_root: data\npaths:\n  models: weights\n')
   self.assertEqual(load(p)['paths']['models'],str(Path(d)/'weights'))
if __name__=='__main__':unittest.main()

class ResourceTests(unittest.TestCase):
 def test_camera_mismatch(self):
  from evoworld.resources import compatibility
  c={'observation_contract':'side70','resource_checks':{'train_data':{'observation_contract':'center'}}}
  with self.assertRaises(ValueError):compatibility(c,'sft')
 def test_corrupt_weight(self):
  from evoworld.resources import verify
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'weight';p.write_bytes(b'x')
   with self.assertRaises(ValueError):verify({'resources':{'base_model':str(p)},'resource_checks':{'base_model':{'sha256':{'weight':'invalid'}}}})

class InterpreterTests(unittest.TestCase):
 def test_preserve_virtualenv_symlink(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'python-real').write_text('');(root/'venv-python').symlink_to(root/'python-real')
   p=root/'local.yaml';p.write_text('environments:\n  rlinf: venv-python\n')
   self.assertEqual(load(p)['environments']['rlinf'],str(root/'venv-python'))
