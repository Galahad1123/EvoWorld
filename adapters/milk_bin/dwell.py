"""Insert an explicit stationary interval at the start of a motion stage."""
import math
import numpy as np

def prepend_dwell(trajectory,current_pose,dt,seconds=.3):
 if dt<=0 or seconds<0:raise ValueError('Invalid dwell duration/control period')
 n=math.ceil(seconds/dt-1e-9)
 trajectory=np.asarray(trajectory)
 return np.concatenate([np.tile(np.asarray(current_pose,dtype=trajectory.dtype),(n,1)),trajectory],axis=0)
