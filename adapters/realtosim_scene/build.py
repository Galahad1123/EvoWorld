"""Authored generic asset + original realtosim builder/package CLI, without robot coupling."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Gf, Sdf

out = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parents[2]
estimate = json.loads((out / 'image-estimate.json').read_text())
size = np.asarray(estimate['cube_size_m'], dtype=float)
xy = np.asarray(estimate['cube_xy_m'], dtype=float)
color = np.asarray(estimate['cube_color_rgb'], dtype=float)
yaw = float(estimate['cube_yaw_degrees'])
assert size.shape == (3,) and np.isfinite(size).all() and ((.025 <= size) & (size <= .08)).all()
assert xy.shape == (2,) and np.isfinite(xy).all() and abs(xy[0]) < .5 and abs(xy[1]) < .25
assert color.shape == (3,) and np.isfinite(color).all() and ((0 <= color) & (color <= 1)).all()
assert math.isfinite(yaw)

asset_dir = out / 'authored-assets'
asset_dir.mkdir(exist_ok=False)
data = np.load(out / 'source-assets/table-mesh.npz')
vertices, faces = data['vertices'], data['faces']
uv = data['uv'][data['uv_indices']].reshape(-1, 2)
assert np.isfinite(vertices).all() and np.isfinite(uv).all()
shutil.copy2(out / 'source-assets/table.png', asset_dir / 'table.png')
bounds = np.array([vertices.min(0), vertices.max(0)])
table_size = bounds[1] - bounds[0]
top = float(bounds[1, 2])
stage = Usd.Stage.CreateNew(str(asset_dir / 'table.usda'))
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, 'Z')
table = UsdGeom.Xform.Define(stage, '/Table')
stage.SetDefaultPrim(table.GetPrim())
mesh = UsdGeom.Mesh.Define(stage, '/Table/Visual')
mesh.CreatePointsAttr(vertices.tolist())
mesh.CreateFaceVertexCountsAttr([3] * len(faces))
mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
mesh.CreateSubdivisionSchemeAttr('none')
UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(uv.tolist())
mat = UsdShade.Material.Define(stage, '/Table/Material')
surface = UsdShade.Shader.Define(stage, '/Table/Material/Surface')
surface.CreateIdAttr('UsdPreviewSurface')
surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.2)
surface.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.3)
reader = UsdShade.Shader.Define(stage, '/Table/Material/UV')
reader.CreateIdAttr('UsdPrimvarReader_float2')
reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
tex = UsdShade.Shader.Define(stage, '/Table/Material/Texture')
tex.CreateIdAttr('UsdUVTexture')
tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('table.png'))
tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
mat.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), 'surface')
UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
pm.CreateStaticFrictionAttr(.6)
pm.CreateDynamicFrictionAttr(.5)
pm.CreateRestitutionAttr(0.)
proxies = [('Top', [float(table_size[0]), float(table_size[1]), .04], [0., 0., top - .02])]
for i, (x, y) in enumerate([(x, y) for x in [-1, 1] for y in [-1, 1]]):
    proxies.append((f'Leg{i}', [.06, .06, top - .04],
                    [x * float(table_size[0]) * .42, y * float(table_size[1]) * .38, (top - .04) / 2]))
for name, dims, pos in proxies:
    p = UsdGeom.Cube.Define(stage, '/Table/Collision/' + name)
    p.CreateSizeAttr(1.)
    p.AddTranslateOp().Set(Gf.Vec3d(*pos))
    p.AddScaleOp().Set(Gf.Vec3f(*dims))
    p.CreatePurposeAttr(UsdGeom.Tokens.guide)
    p.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(p.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(mat, materialPurpose='physics')
stage.GetRootLayer().Save()

def obj(name, kind, dims, pos, rgb, rigid=False):
    return {'id': name, 'kind': kind, 'size_m': dims, 'position_m': pos,
            'quaternion_wxyz': [1, 0, 0, 0],
            'material': {'base_color': rgb, 'roughness': .6, 'metallic': 0., 'status': 'estimated appearance; designed response'},
            'texture': {'status': 'uniform observed color; no texture detail visible', 'path': None},
            'physics': {'rigid': rigid, 'mass_kg': .08 if rigid else 0., 'collision': True,
                        'static_friction': .6, 'dynamic_friction': .5, 'restitution': 0.,
                        'deformable': False, 'status': 'designed defaults; NOT physically calibrated'},
            'evidence': {'source': 'gpt image estimate' if rigid else 'designed support',
                         'size_status': 'image_estimated' if rigid else 'designed',
                         'placement_status': 'uncalibrated visual estimate + designed support constraint' if rigid else 'designed'}}

table_obj = obj('table', 'usd', table_size.tolist(), [0., 0., 0.], [.2, .3, .8])
table_obj.update(asset='authored-assets/table.usda', prim_path='/Table')
table_obj['material'] = {'status': 'reused original SAGE PBR values', 'roughness': .2, 'metallic': .3}
table_obj['texture'] = {'status': 'reused original PNG and face-varying UVs', 'path': 'authored-assets/table.png'}
table_obj['evidence'] = {'source': 'official SAGE-10k 54a67a1f; source-assets/provenance.json',
                         'size_status': 'source-asset metric bounds; not real-world measured',
                         'placement_status': 'designed canonical origin',
                         'collision_status': 'designed 5-box approximation; visual geometry unchanged'}
block = obj('block', 'box', size.tolist(), [float(xy[0]), float(xy[1]), top + float(size[2]) / 2 + .001], color.tolist(), True)
angle = math.radians(yaw) / 2
block['quaternion_wxyz'] = [math.cos(angle), 0., 0., math.sin(angle)]
spec = {'units': 'm', 'up_axis': 'Z', 'dome_intensity': 800,
        'scale_anchor': {'status': 'source_asset_reference', 'description': 'Known synthetic table dimensions; no real-world calibration'},
        'objects': [obj('ground', 'box', [3., 3., .05], [0., 0., -.025], [.18, .18, .18]), table_obj, block]}
(out / 'scene.json').write_text(json.dumps(spec, indent=2))
(out / 'route_plan.json').write_text(json.dumps({
    'scope': ['table', 'block', 'designed support ground'], 'input_kind': 'synthetic, uncalibrated rendered views',
    'routes': {'table': 'reuse visual mesh/texture; static collision approximation', 'block': 'GPT image estimate -> analytic box', 'ground': 'designed support'},
    'uncertainties': estimate['uncertainties'], 'physics_status': 'designed uncalibrated mass/friction',
    'not_reconstructed': ['robot', 'chairs', 'walls'], 'future_robot': 'SAGE Franka; no robot is run in this asset test',
    'table_collision_proxies': proxies}, indent=2))

for command, args in [('build-scene', [out / 'scene.json', out / 'built']),
                      ('package-assets', [out / 'built/scene.usda', out / 'packaged'])]:
    p = subprocess.run([sys.executable, '-m', 'g1_realtosim', command, *map(str, args)],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    (out / (command + '.log')).write_text(p.stdout)
    assert p.returncode == 0, (command, p.stdout)
moved = out / 'relocated/scene-package'
shutil.copytree(out / 'packaged', moved)
shutil.make_archive(str(out / 'portable-scene'), 'zip', root_dir=out / 'packaged')
(out / 'build-report.json').write_text(json.dumps({'passed': True, 'estimate': estimate, 'table_top_m': top,
    'source_table_vertices': len(vertices), 'source_table_faces': len(faces),
    'block_size_m': size.tolist(), 'source_texture_sha256': hashlib.sha256((asset_dir / 'table.png').read_bytes()).hexdigest(),
    'changed_upstream': False, 'portable_entry': 'relocated/scene-package/scene.usda'}, indent=2))
print('PHASE5_BUILD_PACKAGE_COMPLETE', out)
