import json,sys
from pathlib import Path
import numpy as np
import h5py
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from read_trajectory import read


def fixture(path,done=True):
    n=120
    p=np.zeros((n+1,3));p[:,2]=.82;p[30:50,2]=.95;p[60:,0]=.2
    q=np.zeros((n+1,9));q[:,-2:]=.04
    quat=np.zeros((n+1,4));quat[:,0]=1
    target=np.array([.2,0,.82])
    arrays={'joint_pos':q,'joint_vel':np.zeros_like(q),'object_pos':p,'object_quat':quat,
            'object_vel':np.zeros_like(p),'ee_pos':np.zeros_like(p),'ee_quat':quat,
            'target_pos':np.tile(target,(n+1,1))}
    with h5py.File(path,'w') as f:
        data=f.create_group('data');data.attrs['total']=n
        data.attrs['env_args']=json.dumps({'schema':'evoworld.franka_joint_pos_vel.v1','target_world_m':target.tolist(),'metrics':{'success':done,'controller_done':done}})
        d=data.create_group('demo_0');d.attrs['num_samples']=n;d.attrs['success']=done
        d['actions']=np.zeros((n,18));d['rewards']=-np.linalg.norm(p[1:]-target,axis=1)
        flags=np.zeros(n,dtype='u1');flags[-1]=1
        d['dones']=flags;d['terminated']=flags if done else np.zeros(n,dtype='u1');d['truncated']=np.zeros(n,dtype='u1') if done else flags
        d['controller_event']=np.zeros(n,dtype=int);d['time']=np.arange(n)/60
        for k,a in arrays.items():d['obs/'+k]=a[:-1];d['next_obs/'+k]=a[1:]
    return path


def test_valid_episode_and_unfinished_controller(tmp_path):
    assert read(fixture(tmp_path/'ok.h5'))['episode_success']
    assert not read(fixture(tmp_path/'unfinished.h5',done=False))['episode_success']

@pytest.mark.parametrize('bad',['empty_time','final_quat','both_flags','wrong_termination','short_reward','nonfinite_time','wrong_target','shifted_time'])
def test_reader_rejects_malformed_episode(tmp_path,bad):
    path=fixture(tmp_path/'bad.h5')
    with h5py.File(path,'a') as f:
        d=f['data/demo_0']
        if bad=='empty_time':del d['time'];d['time']=np.array([])
        if bad=='final_quat':d['next_obs/object_quat'][-1]=0
        if bad=='both_flags':d['truncated'][-1]=1
        if bad=='wrong_termination':d['terminated'][-1]=0;d['truncated'][-1]=1
        if bad=='short_reward':del d['rewards'];d['rewards']=np.array([0.])
        if bad=='nonfinite_time':d['time'][1]=np.nan
        if bad=='shifted_time':d['time'][:]+=1
        if bad=='wrong_target':d['obs/target_pos'][:]=0;d['next_obs/target_pos'][:]=0
    with pytest.raises((AssertionError,ValueError)):read(path)
