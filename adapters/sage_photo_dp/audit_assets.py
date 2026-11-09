"""Read native photo assets and audit original planner's geometry compatibility."""
import json
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics

def audit(bundle):
 bundle=Path(bundle);spec=json.loads((bundle/'scene_spec.json').read_text());rows=[]
 for obj in spec['objects']:
  stage=Usd.Stage.Open(str(bundle/obj['asset']));colliders=[]
  for prim in stage.Traverse():
   if prim.HasAPI(UsdPhysics.CollisionAPI):
    colliders.append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),
      'enabled':UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False,
      'visibility':str(UsdGeom.Imageable(prim).ComputeVisibility())})
  rows.append({'id':obj['id'],'colliders':colliders,
    'original_mesh_only_planner_compatible':all(x['type']=='Mesh' for x in colliders)})
 return {'assets':rows,'planner_bridge_required':any(not x['original_mesh_only_planner_compatible'] for x in rows),
   'note':'Original SAGE MotionPlanner.init_collision_mesh visits Mesh geometry; native reconstructed Cube colliders require explicit bridge. Do not bypass collisions.'}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('bundle');p.add_argument('--output',required=True);a=p.parse_args()
 r=audit(a.bundle);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
