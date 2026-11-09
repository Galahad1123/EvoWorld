"""Bounded original SAGE robomimic training; dependencies live in envs/sage-dp."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'upstream/sage/robomimic'))
from bootstrap import disable_unused_language_encoder
from training_guards import run_epoch_guarded
disable_unused_language_encoder()
import robomimic.utils.train_utils as TrainUtils
from robomimic.config import config_factory
from robomimic.scripts.train import train

# Original fork independently fits validation action scales. Use only training
# statistics for both splits, so held-out losses measure the same prediction task.
original_load=TrainUtils.load_data_for_training
def load_shared_stats(*args,**kwargs):
    training,validation=original_load(*args,**kwargs)
    if validation is not None:validation.set_action_normalization_stats(training.get_action_normalization_stats())
    return training,validation
TrainUtils.load_data_for_training=load_shared_stats

def build_config(run, epochs=20, name='dp_rgb'):
    run=Path(run).resolve()
    cfg_data=json.loads((ROOT/'upstream/sage/robomimic/robomimic/exps/templates/diffusion_policy.json').read_text())
    c=config_factory('diffusion_policy');c.update(cfg_data)
    with c.unlocked():
        c.experiment.name=name;c.experiment.validate=True
        c.experiment.logging.log_wandb=False;c.experiment.logging.log_tb=True
        c.experiment.epoch_every_n_steps=50;c.experiment.validation_epoch_every_n_steps=5
        c.experiment.save.every_n_epochs=10;c.experiment.save.on_best_validation=False
        c.experiment.save.on_best_rollout_success_rate=False;c.experiment.rollout.enabled=False
        c.train.data=[{'path':str(run/'training.hdf5')}];c.train.output_dir=str(run/'trained')
        c.train.num_data_workers=2;c.train.hdf5_cache_mode='low_dim';c.train.hdf5_load_next_obs=False
        c.train.hdf5_filter_key='train';c.train.hdf5_validation_filter_key='valid'
        c.train.action_config['actions']['normalization']='min_max'
        c.train.batch_size=16;c.train.num_epochs=epochs;c.train.seed=108
        c.train.end_epoch=None  # bundled train.py expects this; its default config omits it
        c.algo.optim_params.policy.learning_rate.warmup_steps=100
        c.algo.ddpm.enabled=False;c.algo.ddim.enabled=True;c.algo.ddim.num_inference_timesteps=10
        c.observation.modalities.obs.rgb=['rgb_left','rgb_right'];c.observation.modalities.obs.depth=[]
        c.observation.encoder.rgb.core_kwargs['backbone_kwargs']['pretrained']=False
    c.lock()
    return c

def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--epochs',type=int,default=20);p.add_argument('--name',default='dp_rgb');args=p.parse_args()
    run=Path(args.run).resolve();c=build_config(run,args.epochs,args.name)
    random.seed(108);np.random.seed(108);torch.manual_seed(108);torch.cuda.manual_seed_all(108)
    torch.backends.cudnn.benchmark=False
    (run/(args.name+'-config.json')).write_text(c.dump())
    original_epoch=TrainUtils.run_epoch;audit=[]
    def audited_epoch(*pos,**kw):
        model=kw['model'];before=next(model.nets.parameters()).detach().flatten()[:64].cpu().clone()
        scheduler_before={k:v.last_epoch for k,v in model.lr_schedulers.items() if v is not None}
        result=run_epoch_guarded(original_epoch,*pos,**kw)
        scheduler_after={k:v.last_epoch for k,v in model.lr_schedulers.items() if v is not None}
        if kw.get('validate',False):assert scheduler_before==scheduler_after
        if any(not np.isfinite(v) for k,v in result[0].items() if isinstance(v,(int,float))):raise ValueError('Nonfinite training/validation metric')
        steps=sorted({int(state['step']) for opt in model.optimizers.values() for state in opt.state.values() if 'step' in state})
        entry={'epoch':kw['epoch'],'validation':kw.get('validate',False),'metrics':result[0],
               'optimizer_steps':steps,'scheduler_before':scheduler_before,'scheduler_after':scheduler_after,'sampled_parameter_changed':not torch.equal(before,next(model.nets.parameters()).detach().flatten()[:64].cpu())}
        audit.append(entry)
        with (run/(args.name+'-audit.jsonl')).open('a') as f:f.write(json.dumps(entry)+'\n')
        return result
    TrainUtils.run_epoch=audited_epoch
    start=time.monotonic();train(c,torch.device('cuda:0'))
    actual_steps=max(audit[-1]['optimizer_steps']);assert actual_steps==args.epochs*50
    assert any(x['sampled_parameter_changed'] for x in audit if not x['validation'])
    assert not any(x['sampled_parameter_changed'] for x in audit if x['validation'])
    report={'wall_seconds':time.monotonic()-start,'optimizer_steps':actual_steps,'seed':108,
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'training_wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'validation_normalization':'training statistics only','peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'complete':True}
    (run/(args.name+'-training-complete.json')).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
