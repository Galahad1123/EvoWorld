"""Robot-base Cartesian metre / spatial rotation-vector radians / gripper.

Delta rotation left-multiplies current rotation in robot-base coordinates.
The gripper convention is +1=open, -1=close, including online decoding.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def absolute_to_delta(current_pose,absolute_action):
 current_pose=np.asarray(current_pose);absolute_action=np.asarray(absolute_action)
 current=Rotation.from_quat(current_pose[...,3:7],scalar_first=True)
 target=Rotation.from_quat(absolute_action[...,3:7],scalar_first=True)
 return np.concatenate([absolute_action[...,:3]-current_pose[...,:3],(target*current.inv()).as_rotvec(),absolute_action[...,-1:]],axis=-1).astype(np.float32)

def delta_to_absolute(current_pose,delta_action):
 current_pose=np.asarray(current_pose);delta_action=np.asarray(delta_action)
 current=Rotation.from_quat(current_pose[...,3:7],scalar_first=True)
 target=Rotation.from_rotvec(delta_action[...,3:6])*current
 gripper=np.where(delta_action[...,-1:]>=0,1.,-1.)
 return np.concatenate([current_pose[...,:3]+delta_action[...,:3],target.as_quat(scalar_first=True),gripper],axis=-1).astype(np.float32)
