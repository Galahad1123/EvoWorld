"""Read native USD visible geometry/materials into the original M2T2 rasterizer contract."""
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Usd,UsdGeom,UsdShade,Gf
from geometry import cube_triangles,triangulate

def atlas_uv(uv,tile,count,size=256):
 # nvdiffrast wrapper flipsV; the sampler then uses normalized*(dimension-1).
 # Map tile endpoints to texel centers so bilinear interpolation cannot bleed.
 uv=np.asarray(uv,dtype=float).copy();uv=np.clip(uv,0,1)
 uv[:,1]=1-(tile*size+(1-uv[:,1])*(size-1))/(count*size-1)
 return uv

def mesh_info(asset,pose,out):
 import trimesh
 from scipy.spatial.transform import Rotation
 stage=Usd.Stage.Open(str(asset));cache=UsdGeom.XformCache();parts=[]
 for prim in stage.Traverse():
  if not prim.IsA(UsdGeom.Gprim) or UsdGeom.Imageable(prim).ComputeVisibility()=='invisible':continue
  if prim.IsA(UsdGeom.Cube):
   v,f=cube_triangles(UsdGeom.Cube(prim).GetSizeAttr().Get());uv=np.full((len(f)*3,2),.5);ft=np.arange(len(uv)).reshape(-1,3)
  elif prim.IsA(UsdGeom.Mesh):
   m=UsdGeom.Mesh(prim);v=np.asarray(m.GetPointsAttr().Get(),float)
   counts=list(m.GetFaceVertexCountsAttr().Get());indices=list(m.GetFaceVertexIndicesAttr().Get());f=triangulate(counts,indices)
   st=UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
   if st and st.HasValue():
    uv=np.asarray(st.ComputeFlattened(),float)
    ft=triangulate(counts,list(range(len(indices)))) if st.GetInterpolation()=='faceVarying' else f
   else:uv=np.full((len(f)*3,2),.5);ft=np.arange(len(uv)).reshape(-1,3)
  else:raise NotImplementedError('Visible geometry '+prim.GetTypeName())
  v=np.array([cache.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(*x)) for x in v])
  color=[.7,.7,.7];texture=None
  mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
  if mat:
   shader=mat.ComputeSurfaceSource()[0]
   if shader:
    inp=shader.GetInput('diffuseColor')
    if inp and inp.Get() is not None:color=list(inp.Get())
    if inp and inp.HasConnectedSource():
     tex=UsdShade.Shader(inp.GetConnectedSource()[0].GetPrim()).GetInput('file').Get()
     if tex:texture=Image.open(tex.resolvedPath or str(Path(asset).parent/tex.path)).convert('RGB').resize((256,256))
  if texture is None:texture=Image.new('RGB',(256,256),tuple(int(np.clip(c,0,1)*255) for c in color))
  parts.append((v,f,uv,ft,texture))
 if not parts:raise ValueError('No visual geometry')
 atlas=Image.new('RGB',(256,256*len(parts)));vs=[];fs=[];uvs=[];fts=[];nv=nu=0
 for i,(v,f,uv,ft,texture) in enumerate(parts):
  atlas.paste(texture,(0,256*i));uv=atlas_uv(uv,i,len(parts))
  vs.append(v);fs.append(f+nv);uvs.append(uv);fts.append(ft+nu);nv+=len(v);nu+=len(uv)
 vertices=np.concatenate(vs);rot=Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix();vertices=vertices@rot.T+np.asarray(pose[:3])
 path=Path(out);path.parent.mkdir(parents=True,exist_ok=True);atlas.save(path)
 return {'mesh':trimesh.Trimesh(vertices,np.concatenate(fs),process=False),
  'texture':{'vts':np.concatenate(uvs),'fts':np.concatenate(fts),'texture_map_path':str(path)},'transform':np.eye(4)}
