"""Opt-in evidence hooks around unchanged native DSRL worker and model methods."""
import json,os,math
from pathlib import Path
from dsrl_evidence import parameter_report,assert_progress,worker_report,digest_tree

def register():
 from rlinf_extension import register as register_task
 register_task()
 from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy as Policy
 from rlinf.models.embodiment.openpi.tasks.dsrl import Pi0DSRL
 if getattr(Policy,'_evoworld_evidence_installed',False):return
 Policy._evoworld_evidence_installed=True
 from dsrl_checkpoint import install_checkpoint_patch
 install_checkpoint_patch(Policy)
 setup=Policy.setup_model_and_optimizer
 def setup_with_fp32_alpha(self,*args,**kwargs):
  result=setup(self,*args,**kwargs)
  from dsrl_precision import prepare_entropy_precision
  prepare_entropy_precision(self)
  return result
 Policy.setup_model_and_optimizer=setup_with_fp32_alpha
 init,train,save,load=Policy.init_worker,Policy.run_training,Policy.save_checkpoint,Policy.load_checkpoint
 def audit_dir(self):
  path=Path(self.cfg.runner.logger.log_path)/'audit';path.mkdir(parents=True,exist_ok=True);return path
 def audited_init(self):
  init(self)
  self._evoworld_initial=parameter_report(self.model)
  assert self._evoworld_initial['parameters']['frozen']==3353433872,self._evoworld_initial
  (audit_dir(self)/f'init-{os.getpid()}.json').write_text(json.dumps(self._evoworld_initial,indent=2))
 def audited_train(self):
  before=parameter_report(self.model)
  alpha_before=digest_tree(self.entropy_temp.state_dict())
  metrics=train(self)
  assert alpha_before!=digest_tree(self.entropy_temp.state_dict()),'Entropy alpha did not update'
  after=parameter_report(self.model)
  assert_progress(before,after)
  assert after['frozen']==self._evoworld_initial['frozen']
  for key,value in metrics.items():
   assert math.isfinite(float(value)),(key,value)
  record={'update_step':self.update_step,'replay_size':len(self.replay_buffer),'metrics':{k:float(v) for k,v in metrics.items()},'frozen_unchanged':True,'actor_changed':True,'critic_changed':True,'alpha_changed':True}
  with (audit_dir(self)/f'updates-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  return metrics
 def audited_save(self,save_base_path,step):
  result=save(self,save_base_path,step)
  report=worker_report(self)
  (Path(save_base_path)/'evoworld-evidence.json').write_text(json.dumps(report,indent=2))
  return result
 def audited_load(self,load_base_path):
  result=load(self,load_base_path)
  expected=json.loads((Path(load_base_path)/'evoworld-evidence.json').read_text());actual=worker_report(self)
  mismatches=[key for key in expected if actual[key]!=expected[key]]
  report={'passed':not mismatches,'mismatches':mismatches,'expected':expected,'actual':actual}
  (audit_dir(self)/f'resume-{os.getpid()}.json').write_text(json.dumps(report,indent=2))
  assert not mismatches,f'DSRL checkpoint restore mismatch: {mismatches}'
  return result
 Policy.init_worker=audited_init;Policy.run_training=audited_train;Policy.save_checkpoint=audited_save;Policy.load_checkpoint=audited_load
 predict=Pi0DSRL.predict_action_batch
 def audited_predict(self,env_obs,mode='eval',**kwargs):
  import torch
  actions,result=predict(self,env_obs,mode=mode,**kwargs)
  noise=result['forward_inputs']['action']
  assert actions.shape==(1,self.action_chunk,7) and noise.shape==(1,20,32),(actions.shape,noise.shape)
  assert torch.isfinite(actions).all() and torch.isfinite(noise).all()
  assert env_obs['main_images'].shape==env_obs['wrist_images'].shape==(1,224,224,3)
  assert env_obs['states'].shape==(1,8)
  path=Path(os.environ.get('EVOWORLD_DSRL_AUDIT_DIR',str(Path(__file__).resolve().parents[2]/'experiments/phase8-milk-bin-20261108/dsrl-smoke/audit')))
  path.mkdir(parents=True,exist_ok=True)
  with (path/f'rollout-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps({'mode':mode,'actions_shape':list(actions.shape),'noise_shape':list(noise.shape),'finite':True,'dual_rgb':True,'action_abs_max':float(actions.abs().max()),'noise_std':float(noise.float().std())})+'\n')
  preview=path/f'rollout-{os.getpid()}-dual-rgb.png'
  if not preview.exists():
   import numpy as np
   from PIL import Image
   Image.fromarray(np.concatenate([env_obs[k][0].cpu().numpy() for k in ['main_images','wrist_images']],axis=1)).save(preview)
  return actions,result
 Pi0DSRL.predict_action_batch=audited_predict
