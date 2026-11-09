"""Online photo-scene success criteria, matching accepted demonstration gates."""
import numpy as np
from scipy.spatial.transform import Rotation

def physical_result(states,roots,z0,goal,corners,table_z,fingers):
 states=np.asarray(states);roots=np.asarray(roots)
 if states.ndim!=2 or states.shape[1]!=13 or roots.shape!=states.shape or len(states)<25:
  raise ValueError('Need aligned full-state traces with at least0.5s tail')
 if not np.isfinite(states).all() or not np.isfinite(roots).all():raise ValueError('Nonfinite physical state')
 rot=Rotation.from_quat(roots[-1,[4,5,6,3]]).as_matrix()
 bottom=(np.asarray(corners)@rot.T+roots[-1,:3])[:,2].min()
 m={'lift_m':float(states[:,2].max()-z0),'xy_error_m':float(np.linalg.norm(roots[-1,:2]-goal)),
    'support_error_m':float(bottom-table_z),'tilt_deg':float(np.degrees(np.arccos(np.clip(rot[2,2],-1,1)))),
    'tail_speed_m_s':float(np.linalg.norm(states[-25:,7:10],axis=1).max()),
    'tail_angular_speed_rad_s':float(np.linalg.norm(states[-25:,10:13],axis=1).max()),
    'finger_opening_m':float(np.mean(fingers))}
 c={'lift':m['lift_m']>.03,'placement':m['xy_error_m']<.05,'support':abs(m['support_error_m'])<.015,
    'upright':m['tilt_deg']<15,'stable':m['tail_speed_m_s']<.05,'angular_stable':m['tail_angular_speed_rad_s']<.2,
    'released':m['finger_opening_m']>.025}
 return {'metrics':m,'checks':c,'success':all(c.values()),'failed_checks':[k for k,v in c.items() if not v]}


def validated_action(raw):
 a=np.asarray(raw)
 if a.shape!=(8,) or not np.isfinite(a).all() or np.linalg.norm(a[3:7])<.01 or np.max(np.abs(a[:3]))>2:
  raise ValueError('Invalid policy action shape/value/quaternion')
 a=a.astype('float32').copy();a[3:7]/=np.linalg.norm(a[3:7]);a[7]=1 if a[7]>=0 else -1
 return a
