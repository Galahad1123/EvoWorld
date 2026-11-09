"""Geometric, release and stability predicate shared by expert/RL acceptance.

Inputs are rigid-root states (position, wxyz, linear/angular velocity), measured
finger positions and local collision vertices. Box is static and axis aligned.
Requires every sample in the final one-second window to satisfy all conditions.
"""
import numpy as np
from scipy.spatial.transform import Rotation

def check_window(roots,fingers,vertices,box_xyz,inner_xy,floor_local_z,dt,*,gripper_commands):
 roots=np.asarray(roots);fingers=np.asarray(fingers);vertices=np.asarray(vertices)
 needed=int(np.ceil(1./dt))+1
 if len(roots)<needed or len(fingers)!=len(roots):
  return {'success':False,'reason':'insufficient_one_second_window','required_samples':needed}
 commands=np.asarray(gripper_commands)
 if len(commands)!=len(roots):return {'success':False,'reason':'incomplete_gripper_commands'}
 commands=commands[-needed:]
 roots=roots[-needed:];fingers=fingers[-needed:]
 if not all(np.isfinite(x).all() for x in [roots,fingers,vertices,commands]):return {'success':False,'reason':'nonfinite'}
 if np.any(np.linalg.norm(roots[:,3:7],axis=1)<.99):return {'success':False,'reason':'invalid_quaternion'}
 rotations=Rotation.from_quat(roots[:,[4,5,6,3]]).as_matrix()
 world=np.einsum('tij,vj->tvi',rotations,vertices)+roots[:,None,:3]
 local=world-np.asarray(box_xyz)[None,None,:]
 xy_margin=np.asarray(inner_xy)/2-np.abs(local[:,:,:2])
 support_error=local[:,:,2].min(axis=1)-floor_local_z
 checks={'inside':bool(xy_margin.min()>=-.001),
         'supported':bool(np.abs(support_error).max()<.004),
         'released':bool(fingers.min()>.039 and np.all(commands>0)),
         'stable_linear':bool(np.linalg.norm(roots[:,7:10],axis=1).max()<.03),
         'stable_angular':bool(np.linalg.norm(roots[:,10:13],axis=1).max()<.2)}
 return {'success':all(checks.values()),'checks':checks,'samples':needed,
         'xy_margin_m':float(xy_margin.min()),'max_support_error_m':float(np.abs(support_error).max()),
         'max_speed_m_s':float(np.linalg.norm(roots[:,7:10],axis=1).max()),
         'max_angular_speed_rad_s':float(np.linalg.norm(roots[:,10:13],axis=1).max())}
