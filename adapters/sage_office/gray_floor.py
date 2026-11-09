"""Create a reversible light-gray floor variant; verify geometry/physics identity."""
import argparse,collections,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade,UsdUtils

p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
source=a.source.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
manifest=json.loads((source/'manifest.json').read_text());original=source/'office.usdz'
assert hashlib.sha256(original.read_bytes()).hexdigest()==manifest['asset_sha256']
work=out/'unpacked';work.mkdir()
with zipfile.ZipFile(original) as z:
 names=z.namelist();assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
 layer=names[0];assert Path(layer).suffix in ['.usd','.usda','.usdc'];z.extractall(work)
stage=Usd.Stage.Open(str(work/layer));assert stage
floor_path='/Office/floor_room_efdc4606';floor=stage.GetPrimAtPath(floor_path);assert floor.IsA(UsdGeom.Mesh)
material=UsdShade.MaterialBindingAPI(floor).ComputeBoundMaterial()[0]
shader=UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath())+'/PBRShader'))
color=shader.GetInput('diffuseColor');changed_path=str(color.GetAttr().GetPath())
assert color.GetAttr().GetConnections()
color.GetAttr().ClearConnections();color.Set(Gf.Vec3f(.55,.55,.55))
assert shader.GetInput('roughness').Get()==1.0 and shader.GetInput('metallic').Get()==0.0
stage.GetRootLayer().Save()
assert UsdUtils.CreateNewUsdzPackage(str(work/layer),str(out/'office.usdz'))
before=Usd.Stage.Open(str(original));after=Usd.Stage.Open(str(out/'office.usdz'));assert after
def value(v):
 # Packaged asset paths can be repacked while their bytes remain unchanged.
 if isinstance(v,Sdf.AssetPath):return ('asset',Path(v.path).name)
 return str(v)
assert [str(p.GetPath()) for p in before.Traverse()]==[str(p.GetPath()) for p in after.Traverse()]
checked_attributes=0;mesh_count=0;geometry=hashlib.sha256()
for old in before.Traverse():
 new=after.GetPrimAtPath(old.GetPath())
 assert old.GetTypeName()==new.GetTypeName() and old.GetAppliedSchemas()==new.GetAppliedSchemas()
 assert old.GetPropertyNames()==new.GetPropertyNames()
 for attr in old.GetAttributes():
  other=new.GetAttribute(attr.GetName());assert attr.GetTypeName()==other.GetTypeName()
  if str(attr.GetPath())==changed_path:continue
  assert value(attr.Get())==value(other.Get()),str(attr.GetPath())
  assert attr.GetConnections()==other.GetConnections(),str(attr.GetPath())
  checked_attributes+=1
 for rel in old.GetRelationships():assert rel.GetTargets()==new.GetRelationship(rel.GetName()).GetTargets()
 if old.IsA(UsdGeom.Mesh):
  mesh_count+=1
  for key in ['points','faceVertexCounts','faceVertexIndices','normals','primvars:st','extent']:
   left=old.GetAttribute(key).Get();right=new.GetAttribute(key).Get()
   if left is not None:
    arr=np.asarray(left);assert np.array_equal(arr,np.asarray(right));geometry.update(str(old.GetPath()).encode()+key.encode()+arr.tobytes())
  assert UsdGeom.Xformable(old).ComputeLocalToWorldTransform(0)==UsdGeom.Xformable(new).ComputeLocalToWorldTransform(0)
with zipfile.ZipFile(original) as z:
 old_textures=collections.Counter(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.lower().endswith(('.png','.jpg','.jpeg')))
with zipfile.ZipFile(out/'office.usdz') as z:
 new_textures=collections.Counter(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.lower().endswith(('.png','.jpg','.jpeg')))
assert old_textures==new_textures
_,_,unresolved=UsdUtils.ComputeAllDependencies(str(out/'office.usdz'));assert not unresolved
actual=after.GetAttributeAtPath(changed_path);assert not actual.GetConnections()
np.testing.assert_allclose(np.asarray(actual.Get()),[.55,.55,.55],atol=1e-7)
new_sha=hashlib.sha256((out/'office.usdz').read_bytes()).hexdigest()
manifest.update(asset=str(out/'office.usdz'),asset_sha256=new_sha,parent_background=str(source),parent_background_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),material_override={'floor_mesh':floor_path,'changed_attribute':changed_path,'diffuse_linear_rgb':[.55,.55,.55],'roughness':1.0,'metallic':0.0,'description':'User-approved light-gray matte floor. Original black texture disconnected; geometry/collision/physics and other materials unchanged.'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
audit={'passed':True,'mesh_count_unchanged':mesh_count,'unchanged_attributes_checked':checked_attributes,'geometry_sha256':geometry.hexdigest(),'relationships_and_schema_apis_unchanged':True,'packaged_texture_bytes_unchanged':True,'only_changed_attribute':changed_path,'unresolved_dependencies':0,'new_background_sha256':new_sha,'original_package_unchanged':hashlib.sha256(original.read_bytes()).hexdigest()==manifest['parent_background_sha256']}
(out/'material-verification.json').write_text(json.dumps(audit,indent=2));(out/'gray_floor-executed.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(audit),flush=True)
