"""Physical action errors for aligned absolute pose/gripper commands."""
import numpy as np


def action_errors(predicted, reference):
    p=np.asarray(predicted,dtype=np.float64);r=np.asarray(reference,dtype=np.float64)
    if p.shape!=r.shape or p.ndim!=2 or p.shape[1]!=8 or len(p)==0:
        raise ValueError('Expected nonempty aligned Nx8 actions')
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Nonfinite action')
    pn=np.linalg.norm(p[:,3:7],axis=1);rn=np.linalg.norm(r[:,3:7],axis=1)
    if np.any(pn<.01) or np.any(rn<.01):raise ValueError('Degenerate quaternion')
    dot=np.abs(np.sum(p[:,3:7]/pn[:,None]*r[:,3:7]/rn[:,None],axis=1))
    delta=(p[:,:3]-r[:,:3])*100
    return {'position_cm':np.linalg.norm(delta,axis=1),'xyz_error_cm':delta,
            'rotation_deg':np.degrees(2*np.arccos(np.clip(dot,0,1))),
            'gripper_correct':(p[:,7]>=0)==(r[:,7]>=0),
            'reference_open':r[:,7]>=0,'predicted_open':p[:,7]>=0,
            'quaternion_norm':pn}


def summarize(errors, mask=None):
    e={k:np.asarray(v) if mask is None else np.asarray(v)[mask] for k,v in errors.items()}
    n=len(e['position_cm'])
    if not n:return {'frames':0}
    out={'frames':n,'position_mean_cm':float(e['position_cm'].mean()),
         'position_rmse_cm':float(np.sqrt(np.mean(e['position_cm']**2))),
         'position_p95_cm':float(np.percentile(e['position_cm'],95)),
         'xyz_rmse_cm':np.sqrt(np.mean(e['xyz_error_cm']**2,axis=0)).tolist(),
         'rotation_mean_deg':float(e['rotation_deg'].mean()),
         'rotation_p95_deg':float(np.percentile(e['rotation_deg'],95)),
         'gripper_accuracy':float(e['gripper_correct'].mean()),
         'quaternion_norm_min':float(e['quaternion_norm'].min()),
         'quaternion_norm_max':float(e['quaternion_norm'].max())}
    recalls=[]
    for label,mask in [('open',e['reference_open']),('closed',~e['reference_open'])]:
        out['reference_'+label+'_frames']=int(mask.sum())
        recall=float(e['gripper_correct'][mask].mean()) if mask.any() else None
        out['gripper_'+label+'_recall']=recall
        if recall is not None:recalls.append(recall)
    out['gripper_balanced_accuracy']=float(np.mean(recalls)) if len(recalls)==2 else None
    return out


def transition_mask(reference, radius=8):
    ref=np.asarray(reference);mask=np.zeros(len(ref),dtype=bool)
    for t in np.flatnonzero(np.diff(ref[:,7]>=0))+1:
        mask[max(0,t-radius):min(len(ref),t+radius+1)]=True
    return mask


def predict_recorded(policy, observations, mode):
    if mode not in {'chunk8','fresh1'}:raise ValueError('Unknown evaluation mode')
    sizes={len(v) for v in observations.values()}
    if len(sizes)!=1 or next(iter(sizes))==0:raise ValueError('Unaligned observations')
    policy.start_episode();predicted=[]
    for t in range(next(iter(sizes))):
        if mode=='fresh1' and t:policy.start_episode()
        # No reference actions or next_obs are provided to the policy.
        predicted.append(np.asarray(policy({k:v[t] for k,v in observations.items()})).copy())
    return np.asarray(predicted)
