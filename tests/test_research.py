import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from evoworld.config import load
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('research_eval',ROOT/'research/rl/scripts/eval_libero.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class ResearchTests(unittest.TestCase):
 def test_paths_follow_machine_config(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'local.yaml';p.write_text('research:\n  output_root: results\nenvironments:\n  libero: env/bin/python\nupstreams:\n  RLinf: source\nresources:\n  libero_checkpoint: weights\n')
   c=load(p);cmd,env=module.command(c,'run-1','env',2)
   self.assertEqual(env['EVOWORLD_RL_OUTPUT_ROOT'],str(Path(d)/'results'))
   self.assertEqual(env['EVOWORLD_RLINF_SOURCE'],str(Path(d)/'source'))
   self.assertEqual(cmd[0],str(Path(d)/'env/bin/python'))
   self.assertEqual(env['EVOWORLD_LIBERO_CHECKPOINT'],str(Path(d)/'weights'))
 def test_reject_escaping_run_id(self):
  for value in ('','..','.','../old','a/b'):
   with self.assertRaises(ValueError):module.command({},value,'env',2)
 def test_manifest_matches_gitlink(self):
  m=json.loads((ROOT/'manifests/upstreams.json').read_text())['RLinf']
  self.assertNotIn('patch',m);self.assertNotIn('installer_patch',m)
  row=subprocess.check_output(['git','-C',str(ROOT),'ls-files','--stage',m['submodule']],text=True).split()
  self.assertEqual(row[0],'160000');self.assertEqual(row[1],m['commit'])

class LaunchTests(unittest.TestCase):
 def config(self,d):
  return {'storage_root':d,'research':{'output_root':d},'upstreams':{'RLinf':d},
          'resources':{'libero_checkpoint':d},'paths':{'cache':d},'environments':{'libero':'python'}}
 def test_dry_run_creates_no_output(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   with patch.object(module,'load',return_value=self.config(d)),patch('sys.argv',['eval','--run-id','new','--stage','env','--dry-run']),patch('builtins.print'):
    module.main()
   self.assertEqual(list(Path(d).iterdir()),[])
 def test_completed_run_rejected_before_writing_provenance(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'runs/libero/pi05-sft/done';(out/'metrics').mkdir(parents=True)
   (out/'metrics/summary.json').write_text('{}')
   with patch.object(module,'load',return_value=self.config(d)),patch('sys.argv',['eval','--run-id','done','--stage','full']):
    with self.assertRaises(RuntimeError):module.main()
   self.assertFalse((out/'provenance').exists())
