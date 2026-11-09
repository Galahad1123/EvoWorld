"""This photo case only: propose a joint fit using GPT-selected panel bottom corners.

It never changes the measured carton mesh, never labels fitting data as held-out,
and requires an explicit recorded agent selection before replacing scene cameras.
"""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fit import project

p=Path(sys.argv[1]).resolve();destination=Path(sys.argv[2]).resolve()
cams=json.loads((p/'cameras.json').read_text());obs=json.loads((p/'observations.json').read_text())['views']
meta=json.loads((p/'metadata.json').read_text());names=['main','left','right']
keys={'main':'tissue_front','left':'tissue_left_end','right':'tissue_right_end'}
targets={n:[meta['texture_sources'][keys[n]]['polygon_tl_tr_br_bl_px'][i] for i in [3,2]] for n in names}
xs=[]
for n in names:
    m=np.linalg.inv(np.array(cams[n]['world_T_camera_cv']));xs.extend([*Rotation.from_matrix(m[:3,:3]).as_rotvec(),*m[:3,3]])
xs.extend([-.13,.15,.075,.04])
def calc(v,details=False):
    x,y,a,b=v[-4:]
    packet={'main':[[x-a,y-b,.001],[x+a,y-b,.001]],'left':[[x-a,y+b,.001],[x-a,y-b,.001]],'right':[[x+a,y-b,.001],[x+a,y+b,.001]]}
    res=[];report={}
    for i,n in enumerate(names):
        R=Rotation.from_rotvec(v[i*6:i*6+3]).as_matrix();t=v[i*6+3:i*6+6];f=1450.;size=cams[n]['image_size_wh']
        milk=np.array([z['world_xyz'] for z in obs[n]['points']]);pixel=np.array([z['pixel_xy'] for z in obs[n]['points']])
        pred=project(milk,R,t,f,size);q=project(packet[n],R,t,f,size)
        res.extend(((pred-pixel)*2).ravel());res.extend((q-targets[n]).ravel())
        m=np.eye(4);m[:3,:3]=R.T;m[:3,3]=-R.T@t
        report[n]={'world_T_camera_cv':m.tolist(),'focal_length_px':1450.,'image_size_wh':size,
            'focal_constraint':'assumed_shared_1450_not_calibrated','status':'joint fit to milk panel corners and tissue bottom edges; those points are fitting data, not held-out validation',
            'milk_errors_px':np.linalg.norm(pred-pixel,axis=1).tolist(),'tissue_errors_px':np.linalg.norm(q-targets[n],axis=1).tolist()}
    return report if details else res
fit=least_squares(calc,xs,max_nfev=3000,loss='soft_l1',f_scale=4,bounds=([-np.inf]*18+[-.3,.03,.04,.02],[np.inf]*18+[0,.25,.11,.08]))
r={'status':'DIAGNOSTIC proposal only; requires model review before selection; geometry anchor unchanged',
    'camera_assumption':'shared fixed focal1450, centered principal point, zero distortion; no actual camera calibration',
    'fit_points':'12 milk body corner points plus6 tissue wrapper bottom points from GPT texture polygons',
    'packet_center_xy_m':fit.x[-4:-2].tolist(),'packet_width_depth_m':(2*fit.x[-2:]).tolist(),
    'weighted_rms_px':float(np.sqrt(np.mean(np.square(calc(fit.x))))),'cameras':calc(fit.x,True)}
assert not destination.exists(),destination
destination.write_text(json.dumps(r,indent=2));print('JOINT_FIT_PROPOSAL_WRITTEN',destination)
