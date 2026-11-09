"""Reuse official SAGE room meshes/materials as a static office backdrop."""
import argparse,hashlib,json,os,sys,traceback
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--layout',required=True);p.add_argument('--output',required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2];layout=Path(a.layout).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT/'assets/SAGE-10k/kits'))
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
import isaacsim_utils as official
official.start_simulation_app()
try:
 from pxr import Usd,UsdGeom,UsdPhysics,UsdUtils,Gf
 data=json.loads(layout.read_text());assert len(data['rooms'])==1
 room=data['rooms'][0];assert 'office' in room['room_type'].lower()
 meshes=official.export_layout_to_mesh_dict_list_v2(official.dict_to_floor_plan(data),str(layout.parent))
 # Centre the background around the existing task; never move the task or cameras.
 shift=np.array([2.08-room['dimensions']['width']/2,1.97-room['dimensions']['length']/2,0.0])
 lo=np.array([1.10,1.00]);hi=np.array([3.00,2.85])
 objects={o['id']:o for o in room['objects']};removed={};bounds={}
 for name,info in meshes.items():
  b=np.array(info['mesh'].bounds)+shift;bounds[name]=b.tolist()
  if 'ceiling' in name:removed[name]='ceiling omitted for open-top inspection and lighting'
  elif name in objects and np.all(b[1,:2]>lo) and np.all(b[0,:2]<hi):
   removed[name]='mesh AABB intersects reserved robot/table workspace'
 # Removing a support also removes objects placed on that support.
 while True:
  children={n:'support removed: '+o.get('place_id','') for n,o in objects.items() if o.get('place_id') in removed and n not in removed}
  if not children:break
  removed.update(children)
 stage=Usd.Stage.CreateNew(str(out/'office.usda'));root=UsdGeom.Xform.Define(stage,'/Office');stage.SetDefaultPrim(root.GetPrim())
 UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.0)
 retained=[]
 for name,info in meshes.items():
  if name in removed:continue
  mesh=info['mesh'];vertices=np.asarray(mesh.vertices)+shift
  official.convert_mesh_to_usd(stage,'/Office/'+name,vertices,mesh.faces,'none',True,None,texture=info.get('texture'),usd_internal_art_reference_path='/Office/'+name)
  # Official helper authors a placeholder extent; replace it with actual bounds.
  UsdGeom.Mesh(stage.GetPrimAtPath('/Office/'+name)).CreateExtentAttr([Gf.Vec3f(*vertices.min(axis=0)),Gf.Vec3f(*vertices.max(axis=0))])
  retained.append(name)
 stage.GetRootLayer().Save()
 assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
 assert UsdUtils.CreateNewUsdzPackage(str(out/'office.usda'),str(out/'office.usdz'))
 reload=Usd.Stage.Open(str(out/'office.usdz'));assert reload and sum(p.IsA(UsdGeom.Mesh) for p in reload.Traverse())==len(retained)
 manifest={'complete':True,'source_layout':str(layout),'source_layout_sha256':hashlib.sha256(layout.read_bytes()).hexdigest(),'layout_id':data['id'],'room_type':room['room_type'],'building_style':data.get('building_style'),'translation_m':shift.tolist(),'reserved_workspace_xy':[lo.tolist(),hi.tolist()],'retained_meshes':retained,'removed':removed,'world_bounds':bounds,'background_physics':'static triangle-mesh colliders; furniture and doors frozen for backdrop use','task_change':'none; native GPT assets/Franka/task poses remain unchanged','asset':str(out/'office.usdz'),'asset_sha256':hashlib.sha256((out/'office.usdz').read_bytes()).hexdigest()}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));(out/'build_background-executed.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps({k:manifest[k] for k in ['complete','layout_id','translation_m','removed','asset_sha256']}),flush=True)
except BaseException:
 (out/'error.txt').write_text(traceback.format_exc());raise
finally:official.simulation_app.close()
