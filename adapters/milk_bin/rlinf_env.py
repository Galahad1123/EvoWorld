"""Register the task with RLinf's native Isaac Lab worker/chunk/reset adapter."""
from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv
class MilkBinEnv(IsaaclabBaseEnv):
 def _make_env_function(self):
  if self.num_envs!=1:raise ValueError('Milk-bin initial validation supports exactly one environment')
  seed=self.seed;output=self.cfg.init_params.get('output_dir',None)
  def make():
   from sim_task import create_task
   return create_task(seed,output)
  return make
 def _wrap_obs(self,obs):
  p=obs['policy']
  return {'main_images':p['rgb_tabletop'],'wrist_images':p['rgb_wrist'],'states':p['state'],'task_descriptions':[self.task_description]}
