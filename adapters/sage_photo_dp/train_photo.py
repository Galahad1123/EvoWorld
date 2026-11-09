"""Photo tabletop experiment profile; reuse audited original training adapter."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sage_dp'))
import train as original
base_build=original.build_config

def build_config(run,epochs=400,name='photo_dp_20k'):
 c=base_build(run,epochs,name)
 with c.unlocked():
  c.train.batch_size=32
  c.algo.optim_params.policy.learning_rate.warmup_steps=1000
  c.experiment.save.every_n_epochs=40
  c.observation.modalities.obs.rgb=['rgb_tabletop','rgb_wrist']
 assert c.train.num_epochs*c.experiment.epoch_every_n_steps==20000
 c.lock()
 return c

if __name__=='__main__':
 (Path(sys.argv[1])/'train-photo-executed.py').write_bytes(Path(__file__).read_bytes())
 original.build_config=build_config
 if '--epochs' not in sys.argv:sys.argv+=['--epochs','400']
 if '--name' not in sys.argv:sys.argv+=['--name','photo_dp_20k']
 base_epoch=original.TrainUtils.run_epoch
 def lr_epoch(*args,**kwargs):
  model=kwargs['model'];before={k:[g['lr'] for g in v.param_groups] for k,v in model.optimizers.items()}
  result=base_epoch(*args,**kwargs)
  after={k:[g['lr'] for g in v.param_groups] for k,v in model.optimizers.items()}
  with (Path(sys.argv[1])/'learning-rate-audit.jsonl').open('a') as f:
   f.write(json.dumps({'epoch':kwargs['epoch'],'validation':kwargs.get('validate',False),'before':before,'after':after})+'\n')
  return result
 original.TrainUtils.run_epoch=lr_epoch
 original.main()
