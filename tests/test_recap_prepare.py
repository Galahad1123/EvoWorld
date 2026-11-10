import json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class RecapPrepareTests(unittest.TestCase):
 def test_full_rejects_unverified_download(self):
  with tempfile.TemporaryDirectory() as d:
   result=subprocess.run([sys.executable,str(ROOT/'research/rl/scripts/recap_prepare.py'),'--root',d,'--mode','full'],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertIn('must be verified',result.stderr)
 def test_metadata_isolated_and_configuration_immutable(self):
  import yaml
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'third_party').mkdir();(r/'third_party/RLinf').symlink_to(ROOT/'third_party/RLinf',target_is_directory=True)
   data=r/'storage/datasets/recap-libero-task0/libero10_task0_sft';(data/'meta').mkdir(parents=True);(data/'data').mkdir();(data/'videos').mkdir();(data/'meta/info.json').write_text('{}')
   cmd=[sys.executable,str(ROOT/'research/rl/scripts/recap_prepare.py'),'--root',d,'--mode','smoke'];subprocess.run(cmd,check=True,capture_output=True)
   out=r/'research/rl/outputs/scratch/recap-20261110/smoke'
   (out/'data/sft/meta/info.json').write_text('{"test":true}')
   self.assertEqual((data/'meta/info.json').read_text(),'{}');self.assertTrue((out/'data/sft/videos').is_symlink())
   cfg=yaml.safe_load((out/'config/cfg.yaml').read_text());self.assertEqual(cfg['actor']['global_batch_size'],4)
   result=subprocess.run(cmd,capture_output=True,text=True);self.assertNotEqual(result.returncode,0);self.assertIn('refusing configuration overwrite',result.stderr)

class RecapEvalConfigTests(unittest.TestCase):
 def test_standalone_model_resolves_actor_interpolations(self):
  import importlib.util
  try:
   from omegaconf import OmegaConf
  except ImportError:
   self.skipTest("OmegaConf required; run this test in RLinf environment")
  spec=importlib.util.spec_from_file_location('recap_eval',ROOT/'research/rl/scripts/recap_eval.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  cfg=module.model_config(ROOT/'third_party/RLinf',Path('/weights'),'positive')
  resolved=OmegaConf.to_container(cfg,resolve=True)
  self.assertEqual(resolved['openpi']['action_chunk'],5)
  self.assertEqual(resolved['openpi']['num_steps'],10)
  self.assertEqual(resolved['openpi']['action_env_dim'],7)
  self.assertFalse(resolved['openpi']['add_value_head'])

class RecapResumeTests(unittest.TestCase):
 def test_successful_stage_resume_does_not_run_again(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);out=r/'research/rl/outputs/run';(out/'provenance').mkdir(parents=True)
   receipt=out/'provenance/value-execution.json';original='{"exit_code":0,"sentinel":123}';receipt.write_text(original)
   result=subprocess.run([sys.executable,str(ROOT/'research/rl/scripts/recap_stage.py'),'--root',d,'--run',str(out),'--stage','value','--resume'],capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(receipt.read_text(),original)
 def test_pipeline_resume_rejects_changed_budget(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);out=r/'research/rl/outputs/runs/libero/recap/pi05-20k-task0-20261110';(out/'provenance').mkdir(parents=True)
   (out/'pipeline-status.json').write_text('{"stage":"failed"}')
   (out/'provenance/recipe.json').write_text('{"value_steps":8000,"cfg_steps":10000}')
   result=subprocess.run([sys.executable,str(ROOT/'research/rl/scripts/recap_pipeline.py'),'--root',d,'--resume','--cfg-steps','30000'],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertIn('budget differs',result.stderr)

class RecapFullBudgetConfigTests(unittest.TestCase):
 def test_full_budget_uses_native_accumulation(self):
  import yaml
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);(r/'third_party').mkdir();(r/'third_party/RLinf').symlink_to(ROOT/'third_party/RLinf',target_is_directory=True)
   raw=r/'storage/datasets/recap-libero-task0'
   for split in ['sft','train','eval']:
    p=raw/f'libero10_task0_{split}'
    for folder in ['meta','data','videos']:(p/folder).mkdir(parents=True)
    (p/'meta/info.json').write_text('{}')
   (raw/'verified.json').write_text('{}')
   cmd=[sys.executable,str(ROOT/'research/rl/scripts/recap_prepare.py'),'--root',d,'--mode','full','--deadline-unix','123456']
   result=subprocess.run(cmd,capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr)
   out=r/'research/rl/outputs/runs/libero/recap/pi05-20k-task0-20261110'
   cfg=yaml.safe_load((out/'config/cfg.yaml').read_text())
   self.assertEqual(cfg['actor']['global_batch_size'],16)
   self.assertEqual(cfg['actor']['micro_batch_size'],4)
   self.assertEqual(cfg['actor']['optim']['lr_warmup_steps'],1500)
   self.assertEqual(cfg['runner']['deadline_unix'],123456)
   self.assertTrue(cfg['runner']['write_training_status'])
