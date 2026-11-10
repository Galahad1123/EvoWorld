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
