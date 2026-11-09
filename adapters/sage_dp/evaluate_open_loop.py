"""Read-only, recorded-observation evaluation of the original DP rollout wrapper."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from open_loop_metrics import action_errors, summarize, transition_mask, predict_recorded


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def plot_episode(out, demo, ref, predictions, hold, fps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.arange(len(ref))/fps
    fig,axes=plt.subplots(5,1,figsize=(11,13),sharex=True)
    for i,label in enumerate(['x (m)','y (m)','z (m)']):
        axes[i].plot(t,ref[:,i],color='black',label='Expert command')
        for mode,values in predictions.items():
            values=np.asarray(values)
            axes[i].plot(t,values[0,:,i],label=mode+' seed208',alpha=.85)
            axes[i].fill_between(t,values[:,:,i].min(0),values[:,:,i].max(0),alpha=.15)
        axes[i].set_ylabel(label)
    axes[0].legend(ncol=3)
    for mode,values in predictions.items():
        axes[3].plot(t,action_errors(values[0],ref)['rotation_deg'],label=mode)
        axes[4].step(t,(values[0][:,7]>=0).astype(int),where='post',label=mode,alpha=.7)
    axes[3].plot(t,action_errors(hold,ref)['rotation_deg'],label='Hold observed pose',color='grey',alpha=.7)
    axes[3].set_ylabel('Rotation error (deg)');axes[3].legend()
    axes[4].step(t,(ref[:,7]>=0).astype(int),where='post',color='black',label='Expert')
    axes[4].set_ylabel('Gripper open');axes[4].set_xlabel('Expert timeline (s, 50 Hz)');axes[4].legend()
    for ax in axes:ax.grid(alpha=.25)
    fig.suptitle(demo+': recorded-observation action predictions (no simulated execution)\nShading: min/max of 3 seeds; lines: seed208')
    fig.tight_layout();fig.savefig(out/(demo+'-actions.png'),dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run).resolve();checkpoint=Path(args.checkpoint).resolve()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'upstream/sage/robomimic'))
    from bootstrap import disable_unused_language_encoder
    disable_unused_language_encoder()
    import torch
    from robomimic.utils.file_utils import policy_from_checkpoint
    torch.set_num_threads(2);torch.manual_seed(208);np.random.seed(208)
    torch.cuda.reset_peak_memory_stats()
    checkpoint_hash=sha256(checkpoint);data_hash=sha256(run/'training.hdf5')
    original=json.loads((run/'acceptance.json').read_text())
    dataset=json.loads((run/'dataset-report.json').read_text())
    assert checkpoint_hash==original['learned_policy']['checkpoint_sha256']
    assert data_hash==dataset['training_sha256']
    policy,ckpt=policy_from_checkpoint(ckpt_path=str(checkpoint),device=torch.device('cuda:0'))
    cfg=json.loads(ckpt['config']) if isinstance(ckpt['config'],str) else ckpt['config']
    assert cfg['algo']['horizon']=={'observation_horizon':1,'action_horizon':8,'prediction_horizon':16}
    assert cfg['algo']['ddim']['enabled'] and cfg['algo']['ddim']['num_inference_timesteps']==10
    assert policy.action_normalization_stats is not None
    shape=ckpt['shape_metadata'];shape=shape[0] if isinstance(shape,list) else shape
    keys=shape['all_obs_keys'];assert set(keys)=={'ee_pose','joint_qpos','rgb_left','rgb_right'}
    seeds=[208,209,210];modes=['chunk8','fresh1'];records=[];pooled={m:[] for m in modes}
    offsets={i:[] for i in range(8)};baseline_errors=[];baseline_records=[];csv_rows=[]
    snapshot=out/'source';snapshot.mkdir()
    source_hashes={}
    for name in ['evaluate_open_loop.py','open_loop_metrics.py','test_open_loop.py','bootstrap.py']:
        path=Path(__file__).with_name(name);(snapshot/name).write_bytes(path.read_bytes());source_hashes[name]=sha256(path)
    with h5py.File(run/'training.hdf5','r') as f:
        test=[s.decode() for s in f['mask/test'][:]]
        assert set(test)==set(dataset['test']) and len(test)==len(set(test))==2
        assert not set(test)&set(dataset['train']+dataset['valid'])
        for demo in dataset['test']:
            g=f['data/'+demo];ref=g['actions'][:]
            obs={k:g['obs/'+k][:] for k in keys}
            assert all(len(v)==len(ref) and np.isfinite(v).all() for v in obs.values())
            hold=obs['ee_pose'].copy();be=action_errors(hold,ref)
            baseline_errors.append(be);baseline_records.append({'demo':demo,**summarize(be)})
            switch=transition_mask(ref,8);predictions={m:[] for m in modes}
            arrays={'reference_actions':ref,'hold_observed_pose':hold,'transition_mask':switch}
            for mode in modes:
                for seed in seeds:
                    torch.manual_seed(seed);np.random.seed(seed)
                    torch.cuda.synchronize();begin=time.monotonic()
                    with torch.inference_mode():pred=predict_recorded(policy,obs,mode)
                    torch.cuda.synchronize();duration=time.monotonic()-begin
                    errors=action_errors(pred,ref);pooled[mode].append(errors)
                    predictions[mode].append(pred);arrays[f'{mode}_seed{seed}']=pred
                    summary=summarize(errors)
                    records.append({'demo':demo,'mode':mode,'seed':seed,'seconds':duration,
                                    'all':summary,'gripper_switch_window':summarize(errors,switch),
                                    'reference_closed':summarize(errors,~errors['reference_open'])})
                    if mode=='chunk8':
                        for i in range(8):offsets[i].append({k:v[np.arange(len(ref))%8==i] for k,v in errors.items()})
                    for t in range(len(ref)):
                        csv_rows.append([demo,mode,seed,t,t/dataset['fps'],t%8 if mode=='chunk8' else 0,
                                         errors['position_cm'][t],errors['rotation_deg'][t],int(errors['gripper_correct'][t])])
                    print(json.dumps({'demo':demo,'mode':mode,'seed':seed,**summary}),flush=True)
            np.savez_compressed(out/(demo+'-predictions.npz'),**arrays)
            plot_episode(out,demo,ref,predictions,hold,dataset['fps'])
    def merge(items):return {k:np.concatenate([e[k] for e in items]) for k in items[0]}
    aggregate={m:summarize(merge(pooled[m])) for m in modes}
    per_seed={m:{str(s):{key:float(np.average([r['all'][key] for r in records if r['mode']==m and r['seed']==s],
                               weights=[r['all']['frames'] for r in records if r['mode']==m and r['seed']==s]))
                               for key in ['position_mean_cm','rotation_mean_deg','gripper_accuracy']}
                  for s in seeds} for m in modes}
    report={'complete':True,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,
            'dataset_sha256':data_hash,'test_episodes':dataset['test'],'unique_frames':sum(e['frames'] for e in baseline_records),
            'seeds':seeds,'seed_protocol':'Reset RNG to seed before each episode/mode; no per-frame reseeding',
            'methods':{'chunk8':'Original RolloutPolicy queue, 8 actions per prediction; recorded obs only',
                       'fresh1':'Reset queue per recorded frame, score first generated action; diagnostic'},
            'model_input_keys':keys,'horizons':cfg['algo']['horizon'],'ddim_steps':10,
            'metrics':'Denormalized base-frame absolute XYZ; q/-q invariant quaternion geodesic after normalization; gripper>=0 is open',
            'alignment':'obs[t] predicts actions[t]; no next_obs, padding or cross-episode chunks',
            'aggregate':aggregate,'per_seed':per_seed,'episodes':records,
            'chunk_offset':{str(i):summarize(merge(v)) for i,v in offsets.items()},
            'hold_observed_pose':{'description':'Current ee_pose8 including gripper command; persistence diagnostic only',
                                  'aggregate':summarize(merge(baseline_errors)),'episodes':baseline_records},
            'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'wall_seconds':time.monotonic()-started,
            'source_sha256':source_hashes,'limitations':['Only two fixed-scene test episodes',
             'Three seeds are repeated predictions, not additional independent episodes',
             'Open-loop action fit does not measure task success; original closed-loop result stays0/2',
             'Full16-step latent prediction is not scored; only executable8-step chunk or its first action'],
            'no_training':True,'no_simulator':True}
    assert checkpoint_hash==sha256(checkpoint) and data_hash==sha256(run/'training.hdf5')
    with (out/'per-frame-errors.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['demo','mode','seed','frame','time_s','chunk_offset','position_cm','rotation_deg','gripper_correct']);writer.writerows(csv_rows)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('OPEN_LOOP_COMPLETE '+str(out),flush=True)


if __name__=='__main__':main()
