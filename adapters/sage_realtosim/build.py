"""Build a portable cached-photo benchmark using original g1 generic packaging."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,UsdLux,UsdUtils,Sdf,Gf
from g1_realtosim.assets import package
from specs import make_specs,validate_specs

BUILTIN={'OmniPBR.mdl','OmniGlass.mdl','OmniSurface.mdl'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(path,data):Path(path).write_text(json.dumps(data,indent=2))
def dependencies(entry,root):
    root=Path(root).resolve();layers,assets,missing=UsdUtils.ComputeAllDependencies(str(entry))
    if set(map(str,missing))-BUILTIN:raise ValueError('Unresolved assets: '+str(missing))
    paths=[Path(l.realPath).resolve() for l in layers]+[Path(str(p)).resolve() for p in assets]
    if not all(p.is_file() and p.is_relative_to(root) for p in paths):raise ValueError('Escaped/nonlocal dependency')
    for layer in layers:
        for refs in UsdUtils.ExtractExternalReferences(layer.identifier):
            for ref in refs:
                if ref not in BUILTIN and Path(ref).is_absolute():raise ValueError('Absolute asset reference')
    return {'dependency_count':len(paths),'builtin_mdl':list(map(str,missing)),'paths':[str(p.relative_to(root)) for p in paths]}

def assemble(bundle,scene,robot):
    entry=bundle/'scene.usda';stage=Usd.Stage.CreateNew(str(entry));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
    for item in scene['objects']+[{'id':'Robot','asset':robot['asset'],'prim_path':robot['prim_path'],'pose_world':robot['base_pose_world']}]:
        x=UsdGeom.Xform.Define(stage,'/World/'+item['id']);x.GetPrim().GetReferences().AddReference(item['asset'],item['prim_path']);x.ClearXformOpOrder();p=item['pose_world'];x.AddTranslateOp().Set(Gf.Vec3d(*p[:3]));x.AddOrientOp().Set(Gf.Quatf(p[3],Gf.Vec3f(*p[4:7])))
    light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(700)
    phys=UsdPhysics.Scene.Define(stage,'/World/Physics');phys.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));phys.CreateGravityMagnitudeAttr(9.81)
    stage.GetRootLayer().customLayerData={'initialization':'Use loader+robot_spec.json for configured joint reset; bareUSD references preserve source articulated rest geometry'}
    stage.GetRootLayer().Save();return entry

def build(root,output):
    root=Path(root).resolve();out=Path(output).resolve()
    if out.exists():raise FileExistsError('Refusing to overwrite benchmark')
    source=root/'experiments/phase5-photo-f92ngyB1';candidate=source/'round3/build';scene,robot,task=make_specs(json.loads((candidate/'scene.json').read_text()),json.loads((candidate/'metadata.json').read_text()))
    out.mkdir(parents=True);shutil.copytree(candidate/'assets',out/'assets/photo');package(root/'assets/isaac42/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd',out/'robot')
    for name,obj in [('scene_spec',scene),('robot_spec',robot),('task_spec',task)]:write(out/(name+'.json'),obj)
    inputs=out/'provenance/input';shutil.copytree(source/'input',inputs)
    for name in ['acceptance.json','README.md','joint-fit-proposal.json']:shutil.copy2(source/name,out/'provenance'/name)
    shutil.copy2(candidate/'metadata.json',out/'provenance/photo-metadata.json')
    for name in ['request.json','response.json','generated.py','usage.json','review.json','route_plan.json','tool-execution.json']:
        p=source/'round3'/name
        if p.exists():shutil.copy2(p,out/'provenance'/name)
    write(out/'provenance/source_hashes.json',{str(p.relative_to(source)):sha(p) for p in list((source/'input').glob('*'))+list((candidate/'assets').rglob('*')) if p.is_file()})
    entry=assemble(out,scene,robot);report=dependencies(entry,out)
    s=Usd.Stage.Open(str(entry));bodies=[str(p.GetPath()) for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and not str(p.GetPath()).startswith('/World/Robot')];assert set(bodies)=={'/World/milk','/World/tissue'}
    collision=[str(p.GetPath()) for p in s.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)];assert len(collision)>10
    report.update(passed=True,dynamic_asset_roots=bodies,collision_count=len(collision),original_g1_packager_used=True,robot_initialization='runtime loader applies RobotSpec default_q')
    write(out/'build-check.json',report);write(out/'manifest-sha256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(Path(__file__).resolve().parents[2]));parser.add_argument('--output',required=True);args=parser.parse_args();print(json.dumps(build(args.root,args.output),indent=2))
