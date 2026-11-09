"""Preserve a pose path and sample it at slower progress at unchanged control dt."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

def retime_pose(poses,speed=.8):
 p=np.asarray(poses,dtype=float)
 if p.ndim!=2 or p.shape[1]!=7 or not len(p) or not np.isfinite(p).all() or not 0<speed<=1:
  raise ValueError('Expected finite Nx7 pose path and0<speed<=1')
 norms=np.linalg.norm(p[:,3:],axis=1)
 if np.any(norms<.01):raise ValueError('Degenerate quaternion')
 p=p.copy();p[:,3:]/=norms[:,None]
 if len(p)==1:return p
 old=np.arange(len(p));new=np.linspace(0,len(p)-1,int(np.ceil((len(p)-1)/speed))+1)
 xyz=np.stack([np.interp(new,old,p[:,i]) for i in range(3)],axis=1)
 quat=Slerp(old,Rotation.from_quat(p[:,[4,5,6,3]]))(new).as_quat()[:,[3,0,1,2]]
 return np.concatenate([xyz,quat],axis=1)
