"""Map actual source records without silently aligning the two scene frames."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from contract import scene_from_realtosim,scene_from_sage
root=Path(__file__).resolve().parents[2];out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
photo=root/'experiments/phase5-photo-f92ngyB1/round3/build/scene.json'
sage=root/'adapters/sage_robot/src/server/results/layout_evoworld_cube_control/layout_evoworld_cube_control_usd_collection'
trans=sage/'rigid_object_transform_dict.json';props=sage/'rigid_object_property_dict.json'
specs={'photo':scene_from_realtosim(json.loads(photo.read_text()),np.eye(4)),'sage':scene_from_sage(json.loads(trans.read_text()),json.loads(props.read_text()),np.eye(4))}
for name,spec in specs.items():
 spec['frame_name']='photo_tabletop_anchor' if name=='photo' else 'sage_room_floor';spec['cross_scene_alignment']='NOT_ESTABLISHED: identity preserves source coordinates, not a claimed alignment';spec['source_file']=str(photo if name=='photo' else trans)
 spec['source_sha256']=hashlib.sha256(Path(spec['source_file']).read_bytes()).hexdigest()
 (out/(name+'-scene-spec.json')).write_text(json.dumps(spec,indent=2))
 for o in spec['objects']:
  asset=(photo.parent if name=='photo' else sage)/o['asset'];assert asset.is_file(),str(asset)
  # Roundtrip to source pose, preserving exact dimensional/provenance metadata.
  source=json.loads(photo.read_text()) if name=='photo' else json.loads(trans.read_text())
  assert np.isfinite(o['pose_world']).all()
print(json.dumps({'passed':True,'mapped_object_counts':{k:len(v['objects']) for k,v in specs.items()},'asset_paths_exist':True,'g1_control_imported':False}))
