"""Strict reader for the EvoWorld joint-target HDF5 baseline (not SAGE IK actions)."""
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from contract import assess_episode,validate_arrays


def read(path):
    with h5py.File(path,'r') as f:
        data=f['data'];assert list(data)==['demo_0']
        d=data['demo_0'];n=int(d.attrs['num_samples']);assert n==data.attrs['total'] and 61<=n<=1800
        meta=json.loads(data.attrs['env_args']);assert meta['schema']=='evoworld.franka_joint_pos_vel.v1'
        a=d['actions'][:];o=d['obs/joint_pos'][:];nxt=d['next_obs/joint_pos'][:]
        validate_arrays(a,np.concatenate((o,nxt[-1:])),nxt)
        dims={'joint_pos':9,'joint_vel':9,'object_pos':3,'object_quat':4,'object_vel':3,'ee_pos':3,'ee_quat':4,'target_pos':3}
        for key,dim in dims.items():
            x=d['obs/'+key][:];y=d['next_obs/'+key][:]
            assert x.shape==y.shape==(n,dim) and np.isfinite(x).all() and np.isfinite(y).all(),key
            assert np.array_equal(x[1:],y[:-1]),'transition discontinuity: '+key
            if 'quat' in key:
                assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-3) and np.allclose(np.linalg.norm(y,axis=1),1,atol=1e-3),key
            if key=='target_pos':assert np.allclose(x,meta['target_world_m']) and np.allclose(y,meta['target_world_m'])
        def states(key):return np.concatenate((d['obs/'+key][:1],d['next_obs/'+key][:]))
        metrics=assess_episode(states('object_pos'),states('object_vel'),states('joint_pos')[:,-2:],meta['target_world_m'])
        metrics['controller_done']=bool(meta['metrics']['controller_done'])
        metrics['success']=metrics['success'] and metrics['controller_done']
        assert metrics['success']==bool(d.attrs['success'])==meta['metrics']['success']
        for key in ('rewards','dones','terminated','truncated','time','controller_event'):
            assert d[key].shape==(n,) and np.isfinite(d[key][:]).all(),key
        assert np.allclose(d['rewards'][:],-np.linalg.norm(d['next_obs/object_pos'][:]-meta['target_world_m'],axis=1))
        assert d['dones'][:].sum()==1 and d['dones'][-1]==1
        for key in ('dones','terminated','truncated'):
            assert np.isin(d[key][:],[0,1]).all(),key
        assert np.array_equal(d['terminated'][:],d['dones'][:] if metrics['success'] else np.zeros(n))
        assert np.array_equal(d['truncated'][:],np.zeros(n) if metrics['success'] else d['dones'][:])
        assert np.allclose(d['time'][:],np.arange(n)/60)
        assert np.isin(d['controller_event'][:],np.arange(11)).all()
        return {'passed':True,'episode_success':metrics['success'],'episodes':1,'transitions':n,
                'actions_shape':list(a.shape),'observation_keys':list(d['obs']),'metrics':metrics,
                'consumer_scope':'EvoWorld strict h5py reader; SAGE hierarchy only, NOT original IK-action/training compatibility'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('path',type=Path);args=p.parse_args()
    result=read(args.path);print(json.dumps(result,indent=2))
    args.path.with_name('readback.json').write_text(json.dumps(result,indent=2))
