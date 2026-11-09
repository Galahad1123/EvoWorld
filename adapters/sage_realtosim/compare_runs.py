"""Compare real independent processes without claiming stochastic pixels/points are identical."""
import argparse,json
from pathlib import Path
import numpy as np
from readback import verify

def compare(first,second):
    first,second=Path(first),Path(second);verify(first);verify(second)
    a=json.loads((first/'inputs.json').read_text());b=json.loads((second/'inputs.json').read_text())
    assert a['bundle']!=b['bundle'] and a['manifest_sha256']==b['manifest_sha256']
    for key in ['scene','robot','task']:assert a[key]==b[key]
    x=json.loads((first/'trace.json').read_text());y=json.loads((second/'trace.json').read_text());errors={}
    for key in ['q','hand_base','robot_base','action']:
        delta=float(np.max(np.abs(np.array([r[key] for r in x])-np.array([r[key] for r in y]))));assert delta<1e-5;errors[key]=delta
    for name in ['milk','tissue']:
        delta=float(np.max(np.abs(np.array([r['objects'][name] for r in x])-np.array([r['objects'][name] for r in y]))));assert delta<1e-5;errors[name+'_state']=delta
    return {'passed':True,'first_bundle':a['bundle'],'relocated_bundle':b['bundle'],'identical_spec_and_asset_manifest':True,'maximum_absolute_state_differences':errors,'images_and_random_points':'not required to be bit-identical','scope':'independent SAGE process reinitialization from relocated canonical specs and native assets'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('first');p.add_argument('second');p.add_argument('--output',required=True);a=p.parse_args();r=compare(a.first,a.second);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
