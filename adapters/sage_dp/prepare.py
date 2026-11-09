"""Validate raw SAGE data and produce a training copy with episode-level masks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import h5py
import numpy as np


def split_episodes(names, seed):
    if len(names) < 6 or len(set(names)) != len(names):
        raise ValueError('Need at least six unique episodes for train, validation and test')
    names = list(sorted(names)); np.random.default_rng(seed).shuffle(names)
    return names[4:], names[:2], names[2:4]


def inspect_data(path):
    records = {}; hashes = set()
    with h5py.File(path, 'r') as f:
        for name in sorted(f['data']):
            g = f['data'][name]; a = g['actions'][:]; n = len(a)
            if a.shape != (n, 8) or n < 16 or not np.isfinite(a).all():
                raise ValueError(f'{name}: invalid actions')
            if not np.allclose(np.linalg.norm(a[:,3:7],axis=-1), 1, atol=.002):
                raise ValueError(f'{name}: nonunit quaternion')
            if not np.isin(a[:,-1], [-1, 1]).all():
                raise ValueError(f'{name}: invalid gripper command')
            digest = hashlib.sha256(a.tobytes()).hexdigest()
            if digest in hashes: raise ValueError('Identical action trajectories must not inflate episode count')
            hashes.add(digest)
            for prefix in ['obs', 'next_obs']:
                for key in ['ee_pose', 'joint_qpos', 'rgb_left', 'rgb_right']:
                    x=g[f'{prefix}/{key}'][:]
                    if x.shape[0]!=n or not np.isfinite(x).all(): raise ValueError(f'{name}/{prefix}/{key}: length/nonfinite')
                    if key.startswith('rgb') and (x.shape[1:]!=(128,128,3) or x.dtype!=np.uint8):
                        raise ValueError('Unexpected RGB schema')
            mismatch = {}
            for key in ['ee_pose','joint_qpos','rgb_left','rgb_right']:
                mismatch[key] = float(np.max(np.abs(g[f'next_obs/{key}'][:-1].astype(float)-g[f'obs/{key}'][1:].astype(float))))
            if any(v > 1e-5 for v in mismatch.values()): raise ValueError(f'{name}: temporal mismatch {mismatch}')
            records[name]={'frames':n,'action_sha256':digest,'action_min':a.min(0).tolist(),'action_max':a.max(0).tolist(),'adjacent_mismatch':mismatch}
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--extra-run',action='append',default=[]);p.add_argument('--expected-episodes',type=int,default=20);args=p.parse_args();run=Path(args.run).resolve()
    sources=[run]+[Path(x).resolve() for x in args.extra_run]
    records={};raw_reports=[];hashes=set()
    dest=run/'training.hdf5'
    if dest.exists(): raise FileExistsError(dest)
    temporary=dest.with_name('training.incomplete-'+uuid.uuid4().hex[:8]+'.hdf5')
    with h5py.File(temporary,'w') as f:
        data=f.create_group('data')
        for src in sources:
            if 'SAGE_ORIGINAL_LOOP_COMPLETE' not in (src/'collection.log').read_text():raise ValueError('incomplete collector run')
            exit_code=(src/'collection-exit-code.txt').read_text().strip()
            # One recorded launcher parser failure occurred after completed collection.
            # Preserve unknown simulator status; accept only that explicit incident,
            # complete marker, independent HDF5 checks and subsequent physical audit.
            if exit_code!='0':
                exception=src/'launcher-exception.json'
                if exit_code!='unknown-wrapper-mutated' or not exception.exists():raise ValueError('collector process failed')
                event=json.loads(exception.read_text())
                if event.get('wrapper_exit_code')!=2 or not event.get('hdf5_reopened_and_validated'):raise ValueError('unverified launcher exception')
            raw=next(src.glob('robomimic_data/*/experts.hdf5')); checked=inspect_data(raw)
            attempts={f"demo_{r['demo']}":r['attempt'] for r in map(json.loads,(src/'demo-attempts.jsonl').read_text().splitlines())}
            with h5py.File(raw,'r') as inp:
                for name,record in checked.items():
                    if record['action_sha256'] in hashes:raise ValueError('duplicate across sources')
                    hashes.add(record['action_sha256'])
                    new=f'demo_{len(records)}';inp.copy(inp['data'][name],data,name=new)
                    records[new]={**record,'source_run':str(src),'source_demo':name,'attempt':attempts[name]}
                for key,val in inp['data'].attrs.items():data.attrs[key]=val
            raw_reports.append({'path':str(raw),'sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'collector_exit_status':exit_code})
        if len(records)!=args.expected_episodes:raise ValueError(f'Expected {args.expected_episodes} successful demos, got {len(records)}')
        data.attrs['total']=sum(r['frames'] for r in records.values())
        train,valid,test=split_episodes(list(records), 108)
        mask=f.require_group('mask')
        for key,values in [('train',train),('valid',valid),('test',test)]:mask[key]=np.asarray(values,dtype='S')
    temporary.rename(dest)
    report={'raw_sources':raw_reports,
            'training_file':str(dest),'training_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'train':train,'valid':valid,'test':test,'split_seed':108,'episodes':records,'fps':50,
            'action_contract':'base-frame absolute panda_hand xyz metres + wxyz + gripper (+1 open/-1 close)',
            'scope':'validation episodes excluded from gradients; shared scene, not scene-generalization test'}
    (run/'dataset-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
