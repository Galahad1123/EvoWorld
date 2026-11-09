"""Independent USD dependency, geometry, scale and physics checks, plus negative fixtures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, Sdf, Gf

def validate(entry):
    entry=Path(entry).resolve();base=entry.parent
    layers,assets,missing=UsdUtils.ComputeAllDependencies(str(entry))
    assert not missing,('Missing dependencies',list(missing))
    resolved=[Path(x.realPath).resolve() for x in layers]+[Path(str(x)).resolve() for x in assets]
    assert all(p.is_file() and p.is_relative_to(base) for p in resolved)
    for layer in layers:
        for group in UsdUtils.ExtractExternalReferences(layer.identifier):
            for ref in group:
                assert not Path(ref).is_absolute(),ref
                assert Path(Sdf.ComputeAssetPathRelativeToLayer(layer,ref)).resolve().is_relative_to(base)
    stage=Usd.Stage.Open(str(entry));assert UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z'
    rigids=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert set(rigids)=={'/World/milk','/World/tissue'},rigids
    meshes=[];colliders=[]
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            m=UsdGeom.Mesh(p);points=np.asarray(m.GetPointsAttr().Get(),float);indices=np.asarray(m.GetFaceVertexIndicesAttr().Get())
            assert len(points)>0 and np.isfinite(points).all() and indices.min()>=0 and indices.max()<len(points)
            counts=np.asarray(m.GetFaceVertexCountsAttr().Get());assert counts.sum()==len(indices) and np.all(counts>=3)
            meshes.append({'path':str(p.GetPath()),'vertices':len(points),'faces':len(counts)})
        if p.HasAPI(UsdPhysics.CollisionAPI):colliders.append(str(p.GetPath()))
    assert len(meshes)>=8,'Expected newly modeled panel/fold/paper mesh structure'
    assert any('/milk/' in m['path'] for m in meshes) and any('/tissue/' in m['path'] for m in meshes)
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    bounds={}
    for name in ['milk','tissue','desk']:
        p=stage.GetPrimAtPath('/World/'+name);assert p
        b=cache.ComputeWorldBound(p).ComputeAlignedRange();bounds[name]=[list(b.GetMin()),list(b.GetMax())]
    # Primary body collision width/depth must retain the measured 55x38mm anchor.
    milk_colliders=[stage.GetPrimAtPath(p) for p in colliders if p.startswith('/World/milk/')]
    sizes=[]
    for p in milk_colliders:
        box=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','guide'],False,True).ComputeWorldBound(p).ComputeAlignedRange()
        sizes.append(np.asarray(box.GetSize()))
    assert any(np.allclose(sorted(x[:2]),[.038,.055],atol=.0002) for x in sizes),('Measured footprint lost',sizes)
    assert .125<=np.ptp(np.asarray(bounds['milk']),axis=0)[2]<=.20,('Carton/straw envelope scale',bounds['milk'])
    masses={p:float(UsdPhysics.MassAPI(stage.GetPrimAtPath(p)).GetMassAttr().Get()) for p in rigids}
    assert all(0<m<2 for m in masses.values())
    textures=[p for p in resolved if p.suffix.lower() in ['.png','.jpg','.jpeg']]
    assert len(textures)>=3,'Observed packaging texture identities must be retained'
    return {'passed':True,'files':[str(p.relative_to(base)) for p in resolved],'mesh_counts':meshes,'collision_prims':colliders,
            'bounds_m':bounds,'rigid_bodies':rigids,'mass_kg_designed':masses,'texture_count':len(textures),
            'anchor_collision_footprint_verified_m':[.055,.038], 'all_dependencies_local_relative':True}

if __name__=='__main__':
    candidate=Path(sys.argv[1]).resolve();entry=candidate/'relocated/scene.usda'
    report=validate(entry);rejected=[]
    with tempfile.TemporaryDirectory(prefix='photo-asset-negative-') as temp:
        temp=Path(temp)
        for case in ['missing_texture','wrong_scale']:
            bad=temp/case;shutil.copytree(entry.parent,bad)
            if case=='missing_texture':
                texture=next(p for p in bad.rglob('*') if p.suffix.lower()=='.png');texture.unlink()
            else:
                stage=Usd.Stage.Open(str(bad/'scene.usda'));p=stage.GetPrimAtPath('/World/milk')
                op=p.GetAttribute('xformOp:scale')
                if op:op.Set(Gf.Vec3f(2,2,2))
                else:UsdGeom.Xformable(p).AddScaleOp().Set(Gf.Vec3f(2,2,2))
                stage.GetRootLayer().Save();del stage
            try:validate(bad/'scene.usda')
            except (AssertionError,RuntimeError):rejected.append(case)
            else:raise AssertionError('Accepted corrupt scene: '+case)
    report['negative_cases_rejected']=rejected
    (candidate/'asset-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['passed','texture_count','bounds_m','negative_cases_rejected']},indent=2))
