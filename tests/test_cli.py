import tempfile,unittest
from pathlib import Path
from evoworld.config import load
from evoworld.cli import command
from evoworld.doctor import check
class DispatchTests(unittest.TestCase):
 def test_command_does_not_execute(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'m.yaml';p.write_text('environments:\n  rlinf: bin/python\nupstreams:\n  RLinf: deps/RLinf\n')
   c=load(p);cmd=command(c,'sft',['runner.max_steps=20000'])
   self.assertEqual(cmd[0],str(Path(d)/'bin/python'))
   self.assertEqual(cmd[-1],'runner.max_steps=20000')
 def test_empty_config_is_not_ready(self):
  c={'environments':{},'resources':{}}
  self.assertTrue(all(not x['prerequisites_available'] for x in check(c).values()))
 def test_formal_rl_requires_baseline(self):
  from evoworld.rl import gate
  with self.assertRaises(ValueError):gate({'resources':{}})
 def test_explicit_sim_experience(self):
  from evoworld.runtime import isaac_experience
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a.kit';b=Path(d)/'b.kit';a.touch();b.touch()
   self.assertEqual(isaac_experience({'isaac_experience':str(a)}),str(a))
   self.assertEqual(isaac_experience({'isaac_experience':str(b)}),str(b))
