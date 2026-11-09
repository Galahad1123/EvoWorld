import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from evoworld.rl import gate,run

class AuthorizationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.ckpt=self.root/'actor';(self.ckpt/'model_state_dict').mkdir(parents=True)
  self.weights=self.ckpt/'model_state_dict/full_weights.pt';self.weights.write_bytes(b'approved-model')
  self.report=self.root/'report.json'
  self.report.write_text(json.dumps({'complete':True,'benchmark_valid':False,'diagnostic_only':True,'checkpoint':str(self.ckpt),'total':20,'successes':4,'success_rate':.2,'max_control_steps':1000,'executed_chunk':10,'episodes':[{'seed':20000+i,'success':i<4,'steps':500 if i<4 else 1000} for i in range(20)]}))
  self.c={'resources':{'checkpoint':str(self.ckpt),'baseline_manifest':str(self.report)},'paths':{'runs':str(self.root)},'rl':{'walltime_seconds':36000,'accepted_baseline':{'reason':'用户明确接受20%基线并授权10小时','report_sha256':self.sha(self.report),'checkpoint_sha256':self.sha(self.weights),'success_rate':.2,'max_control_steps':1000,'executed_chunk':10}}}
 def sha(self,p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def test_explicit_user_acceptance_allows_exact_report(self):
  self.assertEqual(gate(self.c)['successes'],4)
 def test_legacy_gate_still_rejects_unapproved_20_percent(self):
  self.c['rl'].pop('accepted_baseline')
  with self.assertRaises(ValueError):gate(self.c)
 def test_changed_report_rejected(self):
  self.report.write_text(self.report.read_text()+' ')
  with self.assertRaises(ValueError):gate(self.c)
 def test_changed_weights_rejected(self):
  self.weights.write_bytes(b'other-model')
  with self.assertRaises(ValueError):gate(self.c)
 def test_wrong_execution_profile_rejected(self):
  self.c['rl']['accepted_baseline']['executed_chunk']=5
  with self.assertRaises(ValueError):gate(self.c)
 def test_launch_uses_10h_and_matched_profile(self):
  captured={};child=SimpleNamespace(wait=lambda timeout:captured.setdefault('timeout',timeout) and 0,poll=lambda:0)
  def popen(cmd,**kw):captured.update(command=cmd,**kw);return child
  output=self.root/'run';output.mkdir()
  with patch('evoworld.rl.gate',return_value={}),patch('evoworld.rl.time.time',return_value=100.),patch('evoworld.rl.subprocess.Popen',side_effect=popen),patch.dict('sys.modules',{'dsrl_cleanup':SimpleNamespace(cleanup=lambda marker:{'remaining':[]})}):
   self.assertEqual(run(self.c,['python','train.py'],{},output),0)
  self.assertEqual(captured['timeout'],36000)
  self.assertEqual(float(captured['env']['EVOWORLD_DSRL_DEADLINE']),36100)
  for value in ['actor.model.num_action_chunks=10','env.train.max_episode_steps=1000','env.eval.max_episode_steps=1000','env.train.max_steps_per_rollout_epoch=1000','env.eval.max_steps_per_rollout_epoch=1000']:
   self.assertIn(value,captured['command'])
