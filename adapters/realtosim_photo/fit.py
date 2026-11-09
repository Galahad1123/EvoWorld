"""Approximate pinhole fitting to explicitly sourced image landmarks, not sensor calibration."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw

def project(points, rotation, translation, focal, size):
    p=np.asarray(points)@np.asarray(rotation).T+translation
    return p[:,:2]/p[:,2,None]*focal+np.asarray(size)/2

def fit_camera(view):
    points=view['points']
    if len(points)<4: raise ValueError('At least four sourced correspondences are required')
    xyz=np.array([p['world_xyz'] for p in points],float)
    uv=np.array([p['pixel_xy'] for p in points],float)
    size=np.asarray(view['image_size_wh'],float)
    if not all(np.isfinite(a).all() for a in [xyz,uv,size]):raise ValueError('Nonfinite camera data')
    if xyz.shape!=(len(points),3) or uv.shape!=(len(points),2):raise ValueError('Bad correspondence shapes')
    position=np.asarray(view.get('camera_position_m',[0,-.5,.3]),float)
    target=np.asarray(view.get('look_at_m',[0,0,.06]),float)
    forward=target-position;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    rotation=np.array([right,down,forward])
    rv=Rotation.from_matrix(rotation).as_rotvec()
    t=-rotation@position
    base_f=float(view.get('focal_length_px',max(size)))
    fixed=view.get('focal_length_fixed_px')
    if fixed is not None:
        fixed=float(fixed)
        if not np.isfinite(fixed) or fixed<=0:raise ValueError('Invalid explicit focal prior')
    def decode(x):return Rotation.from_rotvec(x[:3]).as_matrix(),x[3:6],fixed if fixed is not None else np.exp(x[6])
    def residual(x):
        R,t,f=decode(x)
        pixels=project(xyz,R,t,f,size)
        depths=(xyz@R.T+t)[:,2]
        # Weak focal prior makes a planar single-panel fit explicit rather than unconstrained.
        return np.r_[(pixels-uv).ravel(), .01*np.log(f/base_f), np.minimum(depths-.02,0)*1000]
    best=None
    for multiplier in [.65,1.,1.7]:
        x=np.r_[rv,t,np.log(base_f*multiplier)]
        fit=least_squares(residual,x,max_nfev=2500,bounds=(np.r_[[-np.inf]*6,np.log(max(size)*.35)],np.r_[[np.inf]*6,np.log(max(size)*4)]),loss='soft_l1',f_scale=3)
        err=np.linalg.norm(residual(fit.x))
        if best is None or err<best[0]:best=(err,fit.x)
    R,t,f=decode(best[1])
    predicted=project(xyz,R,t,f,size)
    assert np.all((xyz@R.T+t)[:,2]>0)
    matrix=np.eye(4);matrix[:3,:3]=R.T;matrix[:3,3]=-R.T@t
    errors=np.linalg.norm(predicted-uv,axis=1)
    return {'world_T_camera_cv':matrix.tolist(),'focal_length_px':float(f),'image_size_wh':size.astype(int).tolist(),
        'focal_constraint':'assumed_fixed_not_calibrated' if fixed is not None else 'fitted_weak_prior_not_calibrated',
        'rms_reprojection_px':float(np.sqrt(np.mean(errors**2))),'max_reprojection_px':float(errors.max()),
        'predicted_pixels':predicted.tolist(),'observed_pixels':uv.tolist(),'labels':[p.get('label','') for p in points],
        'status':'fitted to GPT/manual photo landmarks with centered principal point and zero distortion; not independent sensor calibration',
        'coplanar_landmarks':bool(np.linalg.matrix_rank(xyz-xyz.mean(0),tol=1e-7)<3)}

if __name__=='__main__':
    candidate=Path(sys.argv[1]).resolve();inputs=Path(sys.argv[2]).resolve()
    observations=json.loads((candidate/'observations.json').read_text())
    report={}
    for name,view in observations['views'].items():
        result=fit_camera(view);report[name]=result
        picture=Image.open(inputs/(name+'.jpg')).convert('RGB');draw=ImageDraw.Draw(picture)
        for a,b in zip(result['observed_pixels'],result['predicted_pixels']):
            x,y=a;draw.ellipse((x-5,y-5,x+5,y+5),outline='lime',width=2)
            x,y=b;draw.line((x-6,y,x+6,y),fill='red',width=2);draw.line((x,y-6,x,y+6),fill='red',width=2)
        picture.thumbnail((960,960));picture.save(candidate/(name+'-fit-overlay.png'))
    (candidate/'cameras.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({name:{k:r[k] for k in ['rms_reprojection_px','max_reprojection_px','coplanar_landmarks']} for name,r in report.items()},indent=2))
