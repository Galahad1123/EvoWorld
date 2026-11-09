"""Physical success, independent of collector flags or training losses."""
import numpy as np

def initial_com_height(rows, attempt):
    matches=[r for r in rows if r['attempt']==attempt and r['iteration']==20]
    if len(matches)!=1:raise ValueError('Missing unique COM state before first saved action')
    return float(matches[0]['object'][2])

def success_checks(states, initial_z, goal_xy, finger_positions):
    states=np.asarray(states);goal_xy=np.asarray(goal_xy)
    if states.ndim!=2 or states.shape[1]!=13 or len(states)<25 or not np.isfinite(states).all():
        raise ValueError('Need at least 0.5s of finite 50Hz object states')
    final=states[-1]
    return {'lift':bool(np.max(states[:,2])-initial_z>.03),
            'place_xy':bool(np.linalg.norm(final[:2]-goal_xy)<.08),
            'released':bool(np.asarray(finger_positions).mean()>.025),
            'supported':bool(abs(final[2]-initial_z)<.015),
            'settled':bool(np.max(np.linalg.norm(states[-25:,7:10],axis=1))<.05)}
