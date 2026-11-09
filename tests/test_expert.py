import tempfile,unittest
from pathlib import Path
from evoworld.expert import compiler_environment
class ExpertEnvironmentTests(unittest.TestCase):
 def test_compiler_uses_selected_environment(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);site=root/'site-packages';(site/'nvidia/cu13/include/cccl').mkdir(parents=True)
   c={'expert_site_packages':str(site),'isaac_experience':str(root/'sim/isaacsim/apps/base.kit')}
   e=compiler_environment(c,{'PATH':'/usr/bin'})
   self.assertEqual(e['CUDA_HOME'],str(site/'nvidia/cu13'))
   self.assertTrue(e['PATH'].startswith(str(site/'nvidia/cu13/bin')))
   self.assertIn(str(site/'nvidia/cu13/include/cccl'),e['CPATH'])
   self.assertEqual(e['TORCH_CUDA_ARCH_LIST'],'12.0')
 def test_cuda_runtime_link_is_local_and_idempotent(self):
  from evoworld.expert import prepare_cuda_link
  with tempfile.TemporaryDirectory() as d:
   site=Path(d);lib=site/'nvidia/cu13/lib';lib.mkdir(parents=True);(lib/'libcudart.so.13').write_bytes(b'test')
   prepare_cuda_link(site);prepare_cuda_link(site)
   self.assertTrue((lib/'libcudart.so').is_symlink())
   self.assertEqual((lib/'libcudart.so').readlink(),Path('libcudart.so.13'))
 def test_disabled_gpt_is_explicit(self):
  from evoworld.doctor import check
  report=check({'environments':{},'resources':{},'disabled_components':{'generate':'在5090执行'}})
  self.assertTrue(report['generate']['disabled'])
  self.assertEqual(report['generate']['reason'],'在5090执行')
  self.assertFalse(report['collect']['prerequisites_available'])
 def test_runtime_library_dirs_preserve_inherited_value(self):
  from evoworld.runtime import runtime_environment
  with tempfile.TemporaryDirectory() as d:
   env=runtime_environment({'runtime_library_dirs':[d]},{'LD_LIBRARY_PATH':'/existing'})
   self.assertEqual(env['LD_LIBRARY_PATH'],d+':/existing')
