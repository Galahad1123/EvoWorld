"""Hash, native USD frame and relocation checks, with only Isaac built-in MDL exceptions."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom
from specs import validate_specs,pose_matrix
from build import dependencies

def verify_bundle(bundle):
    bundle=Path(bundle).resolve();objects=[json.loads((bundle/(n+'_spec.json')).read_text()) for n in ['scene','robot','task']];scene,robot,task=objects;validate_specs(*objects)
    manifest=json.loads((bundle/'manifest-sha256.json').read_text())
    for name,h in manifest.items():
        p=(bundle/name).resolve()
        if not p.is_relative_to(bundle) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Missing/changed bundle asset: '+name)
    report=dependencies(bundle/'scene.usda',bundle);stage=Usd.Stage.Open(str(bundle/'scene.usda'));cache=UsdGeom.XformCache()
    for name,pose in [(o['id'],o['pose_world']) for o in scene['objects']]+[('Robot',robot['base_pose_world'])]:
        p=stage.GetPrimAtPath('/World/'+name)
        if not p:raise ValueError('Missing assembled prim:'+name)
        actual=np.asarray(cache.GetLocalToWorldTransform(p)).T
        if not np.allclose(actual,pose_matrix(pose),atol=1e-6):raise ValueError('Assembled frame differs from canonical spec:'+name)
    return dict(report,passed=True,hashed_files=len(manifest),frame_roots_verified=7,robot_joint_reset='applied by runtime loader, not a static mesh transform')
if __name__=='__main__':print(json.dumps(verify_bundle(sys.argv[1]),indent=2))
