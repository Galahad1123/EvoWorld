import unittest
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from collision_bridge import collect_colliders,bridge_planner_class

class BridgeTests(unittest.TestCase):
 def scene(self):
  s=Usd.Stage.CreateInMemory();r=UsdGeom.Xform.Define(s,'/World/envs/env_0/milk')
  r.AddTranslateOp().Set(Gf.Vec3d(2,3,4))
  c=UsdGeom.Cube.Define(s,str(r.GetPath())+'/body');c.CreateSizeAttr(2)
  c.AddScaleOp().Set(Gf.Vec3f(.0275,.019,.065));c.AddTranslateOp().Set(Gf.Vec3d(0,0,1))
  UsdPhysics.CollisionAPI.Apply(c.GetPrim());c.CreateVisibilityAttr('invisible')
  return s,str(r.GetPath()),c
 def test_invisible_scaled_cube(self):
  s,r,c=self.scene();v,f,paths=collect_colliders(s,r,True)
  np.testing.assert_allclose(np.ptp(v,axis=0),[.055,.038,.13],rtol=1e-6)
  self.assertEqual(len(f),12);self.assertEqual(paths,[str(c.GetPath())])
 def test_static_pose_and_multi_shape(self):
  s,r,c=self.scene();d=UsdGeom.Cube.Define(s,r+'/other');d.CreateSizeAttr(.01)
  UsdPhysics.CollisionAPI.Apply(d.GetPrim())
  local,_,_=collect_colliders(s,r,True);world,faces,paths=collect_colliders(s,r,False)
  np.testing.assert_allclose(world,local+[2,3,4]);self.assertEqual(len(paths),2)
  self.assertLess(faces.max(),len(world));self.assertEqual(len(faces),24)
 def test_disabled_and_visual_not_obstacles(self):
  s,r,c=self.scene();UsdPhysics.CollisionAPI(c).CreateCollisionEnabledAttr(False)
  UsdGeom.Cube.Define(s,r+'/visual')
  with self.assertRaises(ValueError):collect_colliders(s,r,True)
 def test_unsupported_collider_fails(self):
  s,r,c=self.scene();sphere=UsdGeom.Sphere.Define(s,r+'/sphere');UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
  with self.assertRaises(NotImplementedError):collect_colliders(s,r,True)
 def test_dynamic_root_scale_rejected(self):
  s,r,c=self.scene();UsdGeom.Xformable(s.GetPrimAtPath(r)).AddScaleOp().Set(Gf.Vec3f(2,1,1))
  with self.assertRaises(ValueError):collect_colliders(s,r,True)
 def test_rotated_root_and_subclass_coverage(self):
  from types import SimpleNamespace
  s,r,c=self.scene()
  UsdGeom.Xformable(s.GetPrimAtPath(r)).AddRotateZOp().Set(90)
  local,_,_=collect_colliders(s,r,True)
  world,_,_=collect_colliders(s,r,False)
  np.testing.assert_allclose(world,local@np.array([[0,1,0],[-1,0,0],[0,0,1]])+[2,3,4],atol=1e-7)
  class Original:
   def init_collision_mesh(self,t=0):
    self.obstacles_mesh={'background':['/World/envs/env_0/wall/mesh'], 'old':[r+'/body']}
  planner=bridge_planner_class(Original,{'milk':{'dynamic':True}})()
  planner.env=SimpleNamespace(scene=SimpleNamespace(stage=s))
  planner.init_collision_mesh()
  self.assertEqual(set(planner.obstacles_mesh),{'background','milk'})
  self.assertEqual(planner.obstacles_mesh['milk'][0],r+'/NativeCollisionMesh')
  np.testing.assert_allclose(planner.obstacles_mesh['milk'][1],local)
 def test_polygon_mesh_fails_closed(self):
  s,r,c=self.scene();m=UsdGeom.Mesh.Define(s,r+'/quad')
  m.CreatePointsAttr([(0,0,0),(1,0,0),(1,1,0),(0,1,0)])
  m.CreateFaceVertexCountsAttr([4]);m.CreateFaceVertexIndicesAttr([0,1,2,3])
  UsdPhysics.CollisionAPI.Apply(m.GetPrim())
  with self.assertRaises(NotImplementedError):collect_colliders(s,r,True)
 def test_probe_planning_failure_raises(self):
  import ast,textwrap
  from pathlib import Path
  tree=ast.parse(Path(__file__).with_name('probe_collision.py').read_text())
  snippet=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='insert' for t in n.targets))
  body=ast.parse(textwrap.dedent(snippet)).body
  guard=next(n for n in body if isinstance(n,ast.If) and any(isinstance(x,ast.Raise) for x in n.body))
  module=ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[]))
  with self.assertRaises(RuntimeError):exec(compile(module,'probe_guard','exec'),{'joint_plan':None})
  exec(compile(module,'probe_guard','exec'),{'joint_plan':object()})
if __name__=='__main__':unittest.main()
