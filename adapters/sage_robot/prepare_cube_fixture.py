"""A deterministic geometric control fixture; no learned grasp/robot output is substituted."""
import copy
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

root = Path(__file__).resolve().parents[2]
results = root / 'adapters/sage_robot/src/server/results'
source = results / 'layout_evoworld_30dd907ee170'
layout_id = 'layout_evoworld_cube_control'
dest = results / layout_id
dest.mkdir(exist_ok=False)
shutil.copytree(source / 'objects', dest / 'objects')
shutil.copytree(source / 'materials', dest / 'materials')
(dest / 'generation').symlink_to('objects')
layout = json.loads((source / (source.name + '.json')).read_text())
layout['id'] = layout_id
room = layout['rooms'][0]
old = room['objects'][-1]
assert old['type'] == 'mug'
cube = copy.deepcopy(old)
cube.update(id=room['id'] + '_cube', type='cube', source_id='control_cube_005',
            description='Deterministic 5 cm solid cube for original SAGE pipeline control test',
            mass=0.08, dimensions=dict(width=.05, length=.05, height=.05),
            position=dict(x=2.32, y=1.97, z=.8017813188675777),
            place_guidance='Control fixture; procedural geometry, not a GPT-generated asset',
            pbr_parameters=dict(metallic=0., roughness=.6))
room['objects'][-1] = cube
mesh = trimesh.creation.box(extents=[.05] * 3)
mesh.apply_translation([0, 0, .025])
asset = dest / 'objects' / cube['source_id']
mesh.export(str(asset) + '.ply')
# Constant texture: UVs do not change geometry or contact surfaces.
uv = np.tile(np.array([[0., 0.], [1., 0.], [1., 1.]], dtype=np.float32), (len(mesh.faces), 1))
with open(str(asset) + '_tex_coords.pkl', 'wb') as stream:
    pickle.dump(dict(vts=uv, fts=np.arange(len(uv)).reshape(-1, 3)), stream)
Image.new('RGB', (32, 32), (210, 55, 25)).save(str(asset) + '_texture.png')
(dest / (layout_id + '.json')).write_text(json.dumps(layout, indent=2))
(dest / 'fixture-provenance.json').write_text(json.dumps({
    'source_layout': source.name, 'kind': 'procedural_cube_control',
    'cube_side_m': .05, 'cube_mass_kg': .08,
    'mesh_sha256': hashlib.sha256(Path(str(asset) + '.ply').read_bytes()).hexdigest(),
    'target': cube['id'], 'place': room['objects'][0]['id'],
    'expected_planar_transfer_m': .24,
}, indent=2))
print(dest)
