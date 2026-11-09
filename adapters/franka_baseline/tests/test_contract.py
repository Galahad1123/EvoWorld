import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contract import assess_episode, validate_arrays, resolve_command


def episode():
    n=121
    pos=np.zeros((n,3));pos[:,2]=.82;pos[30:50,2]=.95
    pos[60:,0]=.2
    vel=np.zeros((n,3));grip=np.full((n,2),.04)
    return pos,vel,grip,np.array([.2,0,.82])


def test_success_requires_lift_release_target_and_settle():
    p,v,g,t=episode()
    assert assess_episode(p,v,g,t)['success']
    p[:,2]=.82
    assert not assess_episode(p,v,g,t)['success']

@pytest.mark.parametrize('kind',['target','closed','speed','nan','short'])
def test_reject_invalid_success(kind):
    p,v,g,t=episode()
    if kind=='target':t[0]=1
    if kind=='closed':g[-61:]=0
    if kind=='speed':v[-1,0]=1
    if kind=='nan':p[20,0]=np.nan
    if kind=='short':p,v,g=p[:30],v[:30],g[:30]
    assert not assess_episode(p,v,g,t)['success']


def test_dataset_alignment_and_finiteness():
    a=np.zeros((120,18));obs=np.zeros((121,9))
    validate_arrays(a,obs,obs[1:])
    with pytest.raises(ValueError):validate_arrays(a,obs[:-1],obs[1:])
    a[3,2]=np.nan
    with pytest.raises(ValueError):validate_arrays(a,obs,obs[1:])


def test_preserve_velocity_and_hold_sparse_joint_positions():
    previous=np.array([0.,-.5,0.,-2.8,0.,3.,.7,.04,.04])
    a=resolve_command(previous,[.1,.2],[.4,-.3],[0,1])
    assert np.allclose(a[:2],[.1,.2]) and np.array_equal(a[2:9],previous[2:])
    assert np.allclose(a[9:11],[.4,-.3]) and np.count_nonzero(a[11:])==0
    b=resolve_command(previous,[-.04,.09],None,[7,8])
    assert np.array_equal(b[:7],previous[:7]) and np.allclose(b[7:9],[0.,.04])
    assert np.count_nonzero(b[9:])==0
