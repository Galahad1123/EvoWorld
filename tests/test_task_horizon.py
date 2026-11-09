import importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]

class NativeEnv:
 def _init_isaaclab_env(self):
  self.env=SimpleNamespace(step=lambda *a,**kw:('obs','reward','terminated','truncated',{'physical_acceptance':{'success':False,'checks':{'stable_angular':False}}}))
 def _record_metrics(self,reward,terminations,infos):return {'episode':'native metrics'}
 def step(self,actions=None,auto_reset=True):
  obs,reward,term,trunc,discarded_infos=self.env.step(actions)
  return obs,reward,term,trunc,self._record_metrics(reward,term,{})

class HorizonTests(unittest.TestCase):
 def module(self):
  spec=importlib.util.spec_from_file_location('isolated_milk_env',ROOT/'adapters/milk_bin/rlinf_env.py')
  module=importlib.util.module_from_spec(spec)
  with patch.dict(sys.modules,{'rlinf.envs.sim.isaaclab.isaaclab_env':SimpleNamespace(IsaaclabBaseEnv=NativeEnv)}):spec.loader.exec_module(module)
  return module
 def test_configured_horizon_reaches_underlying_task(self):
  module=self.module();env=module.MilkBinEnv();env.num_envs=1;env.seed=12
  env.cfg=SimpleNamespace(max_episode_steps=1000,init_params={'output_dir':'diagnostic'})
  calls=[]
  with patch.dict(sys.modules,{'sim_task':SimpleNamespace(create_task=lambda *a,**kw:calls.append((a,kw)))}):env._make_env_function()()
  self.assertEqual(calls,[((12,'diagnostic'),{'max_episode_steps':1000})])
 def test_physical_result_survives_native_wrapper(self):
  env=self.module().MilkBinEnv();result={'success':False,'checks':{'stable_angular':False}}
  env._init_isaaclab_env()
  values=env.step('action',auto_reset=False)
  self.assertEqual(values[:4],('obs','reward','terminated','truncated'))
  self.assertEqual(values[4]['episode'],'native metrics')
  self.assertEqual(values[4]['physical_acceptance'],result)
