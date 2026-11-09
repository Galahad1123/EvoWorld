"""Fresh-process package/geometry/texture checks and a missing-dependency negative case."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from pxr import Usd, UsdUtils, UsdGeom, UsdPhysics, Sdf

out = Path(sys.argv[1]).resolve()
package = out / 'relocated/scene-package'
entry = package / 'scene.usda'
layers, assets, missing = UsdUtils.ComputeAllDependencies(str(entry))
assert not missing, list(missing)
resolved = [Path(x.realPath).resolve() for x in layers] + [Path(str(x)).resolve() for x in assets]
assert len(layers) >= 2 and len(assets) >= 1
assert all(p.is_file() and p.is_relative_to(package) for p in resolved)
for layer in layers:
    for group in UsdUtils.ExtractExternalReferences(layer.identifier):
        for path in group:
            assert not Path(path).is_absolute(), path
            target = Path(Sdf.ComputeAssetPathRelativeToLayer(layer, path)).resolve()
            assert target.is_file() and target.is_relative_to(package), (path, target)
stage = Usd.Stage.Open(str(entry))
assert UsdGeom.GetStageMetersPerUnit(stage) == 1. and UsdGeom.GetStageUpAxis(stage) == 'Z'
rigids = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
colliders = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
assert rigids == ['/World/block'] and len(colliders) == 7, (rigids, colliders)
assert abs(UsdPhysics.MassAPI(stage.GetPrimAtPath('/World/block')).GetMassAttr().Get() - .08) < 1e-6
source = np.load(out / 'source-assets/table-mesh.npz')
mesh = UsdGeom.Mesh(stage.GetPrimAtPath('/World/table/Visual'))
assert np.array_equal(np.asarray(mesh.GetPointsAttr().Get()), source['vertices'])
uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get()
assert np.allclose(np.asarray(uv), source['uv'][source['uv_indices']].reshape(-1, 2))
png = [p for p in resolved if p.suffix == '.png']
assert len(png) == 1
assert hashlib.sha256(png[0].read_bytes()).hexdigest() == hashlib.sha256((out / 'source-assets/table.png').read_bytes()).hexdigest()
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
bounds = {}
for name in ['table', 'block', 'ground']:
    box = cache.ComputeWorldBound(stage.GetPrimAtPath('/World/' + name)).ComputeAlignedRange()
    bounds[name] = [list(box.GetMin()), list(box.GetMax())]
gap = bounds['block'][0][2] - bounds['table'][1][2]
assert abs(gap - .001) < 1e-5
for axis in [0, 1]:
    assert bounds['table'][0][axis] < bounds['block'][0][axis] < bounds['block'][1][axis] < bounds['table'][1][axis]

bad = out / 'negative-missing-texture'
shutil.copytree(package, bad)
(bad / png[0].relative_to(package)).unlink()  # Only this newly created negative fixture.
p = subprocess.run([sys.executable, '-m', 'g1_realtosim', 'package-assets', str(bad / 'scene.usda'), str(out / 'negative-package-output')],
                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
(out / 'missing-texture-negative.log').write_text(p.stdout)
assert p.returncode != 0 and 'Unresolved dependencies' in p.stdout
(out / 'package-check.json').write_text(json.dumps({'passed': True, 'resolved_files': [str(p.relative_to(package)) for p in resolved],
    'all_references_relative_and_inside_relocated_package': True, 'mesh_and_uv_unchanged': True, 'texture_bytes_unchanged': True,
    'rigid_bodies': rigids, 'collision_prims': colliders, 'bounds_m': bounds,
    'initial_block_support_gap_m': gap, 'missing_texture_negative_rejected': True}, indent=2))
print('PHASE5_RELOCATED_PACKAGE_PASS')
