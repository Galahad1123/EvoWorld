"""Evaluate frozen cameras against final rounded geometry; no optimization."""
import json,sys
from pathlib import Path
import numpy as np
from fit import project
b=Path(sys.argv[1]); cams=json.loads((b/'cameras.json').read_text()); obs=json.loads((b/'observations.json').read_text())
x,y,z=-.131,.152,.001;a,bhalf=.072,.045
points={'main':([[x-a,y-bhalf,z],[x+a,y-bhalf,z]],[[463,646],[846,646]]),'left':([[x-a,y+bhalf,z],[x-a,y-bhalf,z]],[[287,1150],[511,1177]]),'right':([[x+a,y-bhalf,z],[x+a,y+bhalf,z]],[[520,657],[665,650]])}
report={'status':'Residuals on fitting data only; no independent calibration or held-out validation','optimized':False,'views':{}}
for name,c in cams.items():
    T=np.linalg.inv(np.array(c['world_T_camera_cv'])); v=obs['views'][name]; milk=v['points']
    sets={'milk':([p['world_xyz'] for p in milk],[p['pixel_xy'] for p in milk]),'tissue_rounded_final':points[name]}
    report['views'][name]={}
    for label,(xyz,uv) in sets.items():
        pred=project(xyz,T[:3,:3],T[:3,3],c['focal_length_px'],c['image_size_wh']);err=np.linalg.norm(pred-np.asarray(uv),axis=1)
        report['views'][name][label]={'errors_px':err.tolist(),'rms_px':float(np.sqrt(np.mean(err**2))),'max_px':float(err.max()),'max_percent_long_side':float(err.max()/max(c['image_size_wh'])*100)}
(b/'camera-fit-final.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
