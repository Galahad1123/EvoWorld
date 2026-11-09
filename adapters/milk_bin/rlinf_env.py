"""Register the task with RLinf's native Isaac Lab worker/chunk/reset adapter."""
from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv
class MilkBinEnv(IsaaclabBaseEnv):
 def _make_env_function(self):
  if self.num_envs!=1:raise ValueError('Milk-bin initial validation supports exactly one environment')
  seed=self.seed;output=self.cfg.init_params.get('output_dir',None);horizon=int(self.cfg.max_episode_steps)
  def make():
   from sim_task import create_task
   return create_task(seed,output,max_episode_steps=horizon)
  return make
 def _wrap_obs(self,obs):
  p=obs['policy']
  return {'main_images':p['rgb_tabletop'],'wrist_images':p['rgb_wrist'],'states':p['state'],'task_descriptions':[self.task_description]}

 def _init_isaaclab_env(self):
  super()._init_isaaclab_env()
  self._last_physical_acceptance=None
  native_step=self.env.step
  def capture(*args,**kwargs):
   result=native_step(*args,**kwargs)
   self._last_physical_acceptance=result[4].get('physical_acceptance')
   return result
  # self.env is the subprocess proxy, not the task object. Capture returned info
  # before the native wrapper discards it; do not read child-process attributes.
  self.env.step=capture

 def _record_metrics(self,step_reward,terminations,infos):
  infos=super()._record_metrics(step_reward,terminations,infos)
  # The native step discards task infos. Attach before any automatic reset.
  infos['physical_acceptance']=self._last_physical_acceptance
  return infos
