"""Pure numerical acceptance; controller completion alone never implies success."""
import numpy as np


def assess_episode(positions, velocities, fingers, target):
    p,v,g,t=map(lambda x:np.asarray(x,dtype=float),(positions,velocities,fingers,target))
    valid=(p.ndim==2 and p.shape[1:]==(3,) and len(p)>=61 and v.shape==p.shape
           and g.shape==(len(p),2) and t.shape==(3,) and all(np.isfinite(x).all() for x in (p,v,g,t)))
    if not valid:return {'success':False,'reason':'invalid or insufficient samples'}
    lift=float(p[:,2].max()-p[0,2]);errors=np.linalg.norm(p[-61:]-t,axis=1)
    speed=np.linalg.norm(v[-61:],axis=1)
    released=bool((g[-61:]>.025).all())
    stable=bool(speed.max()<.03 and np.linalg.norm(np.diff(p[-61:],axis=0),axis=1).max()<.002)
    success=lift>=.05 and errors.max()<=.03 and released and stable
    return {'success':bool(success),'lift_m':lift,'final_target_error_m':float(errors[-1]),
            'tail_max_target_error_m':float(errors.max()),'tail_max_speed_m_s':float(speed.max()),
            'released':released,'stable':stable,'tail_samples':61}


def validate_arrays(actions, observations, next_observations):
    a,o,n=map(np.asarray,(actions,observations,next_observations))
    if a.ndim!=2 or a.shape[1]!=18 or o.shape!=(len(a)+1,9) or n.shape!=(len(a),9):
        raise ValueError('Expected N x 18 joint position/velocity targets, N+1 observations, N next observations')
    if not all(np.isfinite(x).all() for x in (a,o,n)):
        raise ValueError('Non-finite trajectory')
    if not np.array_equal(o[1:],n):raise ValueError('Misaligned next observations')


def resolve_command(previous_positions, positions, velocities, indices):
    """Preserve both RMP targets; hold absent positions and zero absent feed-forward velocities."""
    q=np.asarray(previous_positions,dtype=float).copy();v=np.zeros(9)
    for values,dest in ((positions,q),(velocities,v)):
        if values is not None:
            ids=range(len(values)) if indices is None else indices
            for i,value in zip(ids,values):
                if value is not None:dest[int(i)]=float(value)
    q[7:]=np.clip(q[7:],0.,.04)
    return np.concatenate((q,v))
