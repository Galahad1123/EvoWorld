"""Bound native DSRL at completed-round boundaries and retain completed checkpoints."""
import json,re,shutil,time
from pathlib import Path
class BudgetReached(Exception):pass

def retain_completed(root,keep=2):
 root=Path(root).resolve()
 completed=sorted([p for p in root.iterdir() if re.fullmatch(r'global_step_\d+',p.name) and not p.is_symlink() and (p/'completed.json').is_file()],key=lambda p:int(p.name.split('_')[-1]))
 for p in completed[:-keep]:
  assert p.resolve().parent==root
  shutil.rmtree(p)

def install_budget(Runner,deadline,clock=time.time,retention=2):
 original_run,original_log,original_save=Runner.run,Runner._log_step_metrics,Runner._save_checkpoint
 def save(self):
  result=original_save(self)
  if retention is not None:
   root=Path(self.cfg.runner.logger.log_path)/self.cfg.runner.logger.experiment_name/'checkpoints'
   checkpoint=root/f'global_step_{self.global_step}'
   assert (checkpoint/'actor/evoworld-evidence.json').is_file()
   (checkpoint/'completed.json').write_text(json.dumps({'global_step':self.global_step,'time':clock()}))
   retain_completed(root,retention)
  self._evoworld_last_saved_step=getattr(self,'global_step',None)
  return result
 def log(self,*args,**kwargs):
  result=original_log(self,*args,**kwargs)
  if clock()>=deadline-300:
   if not hasattr(self,'_evoworld_last_saved_step') or self._evoworld_last_saved_step!=getattr(self,'global_step',None):self._save_checkpoint()
   raise BudgetReached()
  return result
 def run(self,*args,**kwargs):
  try:return original_run(self,*args,**kwargs)
  except BudgetReached:
   self._finish_run()
   if retention is not None:
    path=Path(self.cfg.runner.logger.log_path)/'budget-complete.json'
    path.write_text(json.dumps({'reason':'four_hour_budget','global_step':self.global_step,'finished_at':clock(),'deadline':deadline},indent=2))
 Runner.run=run;Runner._log_step_metrics=log;Runner._save_checkpoint=save
