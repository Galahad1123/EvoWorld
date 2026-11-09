"""Native photo-asset collision geometry for the original SAGE MotionPlanner.

Does not modify USD or replace the planner. Explicit object IDs select assets
whose CollisionAPI primitives supersede the old visual-mesh heuristic.
"""
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from geometry import cube_triangles,triangulate


def collect_colliders(stage,root_path,dynamic):
 root=stage.GetPrimAtPath(root_path)
 if not root:raise ValueError('Missing collision root '+root_path)
 cache=UsdGeom.XformCache();root_world=np.asarray(cache.GetLocalToWorldTransform(root))
 if dynamic and not np.allclose(root_world[:3,:3]@root_world[:3,:3].T,np.eye(3),atol=1e-6):
  raise ValueError('Dynamic root scaling/shear unsupported; bake asset dimensions first')
 origin=np.linalg.inv(root_world) if dynamic else np.eye(4)
 vertices=[];faces=[];paths=[];offset=0
 for prim in Usd.PrimRange(root):
  if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
  if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
  if prim.IsA(UsdGeom.Cube):v,f=cube_triangles(UsdGeom.Cube(prim).GetSizeAttr().Get())
  elif prim.IsA(UsdGeom.Mesh):
   approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
   if approximation not in (None,'none'):raise NotImplementedError('Approximate mesh collision '+str(approximation))
   m=UsdGeom.Mesh(prim);v=np.asarray(m.GetPointsAttr().Get(),dtype=float)
   counts=list(m.GetFaceVertexCountsAttr().Get())
   if any(n!=3 for n in counts) or m.GetHoleIndicesAttr().Get():
    raise NotImplementedError('Only triangle Mesh colliders without holes are supported')
   if m.GetOrientationAttr().Get()=='leftHanded':raise NotImplementedError('Left-handed mesh collider')
   f=triangulate(counts,list(m.GetFaceVertexIndicesAttr().Get()))
  else:raise NotImplementedError('Unsupported enabled collider '+prim.GetTypeName()+' '+str(prim.GetPath()))
  transform=np.asarray(cache.GetLocalToWorldTransform(prim))@origin
  if abs(np.linalg.det(transform[:3,:3]))<1e-12:raise ValueError('Degenerate collider transform')
  v=(np.c_[v,np.ones(len(v))]@transform)[:,:3]
  if np.linalg.det(transform[:3,:3])<0:f=f[:,::-1]
  if not np.isfinite(v).all():raise ValueError('Nonfinite collision geometry')
  vertices.append(v);faces.append(f+offset);offset+=len(v);paths.append(str(prim.GetPath()))
 if not paths:raise ValueError('No enabled collision geometry under '+root_path)
 return np.concatenate(vertices),np.concatenate(faces),paths


def bridge_planner_class(original_class,object_specs):
 """Return an opt-in subclass; instantiate it instead of the original class.

 object_specs maps scene object IDs to {dynamic: bool}. Dynamic geometry stays
 in rigid-root coordinates; original add_obstacle reads its live PhysX pose.
 Static geometry is baked into world coordinates because original add_obstacle
 uses identity for AssetBase. Original background meshes remain untouched.
 """
 class NativeCollisionPlanner(original_class):
  def init_collision_mesh(self,timecode=0):
   super().init_collision_mesh(timecode)
   self.native_collision_report=[]
   for name,spec in object_specs.items():
    root='/World/envs/env_0/'+name
    v,f,paths=collect_colliders(self.env.scene.stage,root,spec['dynamic'])
    for key,value in list(self.obstacles_mesh.items()):
     if value is not None and (value[0]==root or value[0].startswith(root+'/')):del self.obstacles_mesh[key]
    # Original lookup takes object ID from path segment4. One merged mesh per
    # object prevents multi-part shapes overwriting each other by parent name.
    self.obstacles_mesh[name]=[root+'/NativeCollisionMesh',v,f,[1,1,1]]
    self.native_collision_report.append({'object':name,'dynamic':spec['dynamic'],
      'collider_paths':paths,'vertices':len(v),'triangles':len(f),
      'bounds':np.array([v.min(0),v.max(0)]).tolist()})
 return NativeCollisionPlanner
