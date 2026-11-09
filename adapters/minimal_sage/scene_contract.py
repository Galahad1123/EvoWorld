"""Bounded new layouts using unchanged, explicitly attributed SAGE-10k assets."""
import copy
import hashlib
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

DATASET_REVISION = 'd425de49ff445fd3903d790a11461d23ffd0c7dd'
SAGE_COMMIT = 'fe3966013a495dc1642710de93a83ddf66e534a0'


def catalog_from_fixture(fixture: Path) -> dict:
    data = json.loads((fixture / 'layout_84b703fb.json').read_text())
    catalog = {}
    for obj in data['rooms'][0]['objects']:
        kind = obj['type'].split('_')[0]
        if kind not in {'table', 'chair', 'mug'}:
            continue
        asset_id = obj['source_id']
        if asset_id in catalog:
            continue
        for suffix in ['.ply', '_texture.png']:
            if not (fixture / 'objects' / (asset_id + suffix)).is_file():
                raise ValueError(f'Missing asset file for {asset_id}')
        catalog[asset_id] = {'asset_id': asset_id, 'kind': kind,
            'description': obj['description'], 'dimensions': obj['dimensions'],
            'mass': obj.get('mass', 1.0), 'pbr_parameters': obj.get('pbr_parameters') or {}, 'source_object_id': obj['id']}
    return catalog


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Coordinates and dimensions must be finite numbers')
    return value


def footprint(obj, catalog):
    dim = catalog[obj['asset_id']]['dimensions']
    w, length = dim['width'], dim['length']
    if int(obj['yaw']) % 180 == 90:
        w, length = length, w
    return (obj['x']-w/2, obj['y']-length/2, obj['x']+w/2, obj['y']+length/2)


def validate_plan(plan: dict, catalog: dict) -> dict:
    if set(plan) != {'title', 'width', 'length', 'height', 'objects'}:
        raise ValueError('Plan requires exactly title,width,length,height,objects')
    if not isinstance(plan['title'], str) or not 1 <= len(plan['title']) <= 200:
        raise ValueError('Title must be 1-200 characters')
    for field in ['width', 'length']:
        if not 3 <= _number(plan[field]) <= 6:
            raise ValueError('Room width/length must be 3-6 meters')
    if not 2.4 <= _number(plan['height']) <= 3.2:
        raise ValueError('Room height must be 2.4-3.2 meters')
    objects = plan['objects']
    if not isinstance(objects, list) or len(objects) != 4:
        raise ValueError('Exactly one table, two chairs and one mug are required')
    names = set()
    kinds = []
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {'name','asset_id','x','y','z','yaw','support'}:
            raise ValueError('Each object requires name,asset_id,x,y,z,yaw,support')
        name = obj['name']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,31}', name) or name in names:
            raise ValueError('Object names must be unique safe identifiers')
        names.add(name)
        if not isinstance(obj['asset_id'], str) or obj['asset_id'] not in catalog:
            raise ValueError('Unknown asset ID')
        kinds.append(catalog[obj['asset_id']]['kind'])
        for field in ['x','y','z','yaw']:
            _number(obj[field])
        if obj['yaw'] not in [0,90,180,270]:
            raise ValueError('Yaw must be 0,90,180,270 degrees')
        box = footprint(obj, catalog)
        if box[0] < .08 or box[1] < .08 or box[2] > plan['width']-.08 or box[3] > plan['length']-.08:
            raise ValueError(f'{name} crosses the room margin')
        if obj['z'] < 0 or obj['z'] + catalog[obj['asset_id']]['dimensions']['height'] >= plan['height']:
            raise ValueError(f'{name} crosses the vertical room bounds')
    if Counter(kinds) != {'table':1,'chair':2,'mug':1}:
        raise ValueError('Scene must contain one table, two chairs and one mug')
    table = next(o for o in objects if catalog[o['asset_id']]['kind']=='table')
    tb = footprint(table, catalog)
    for obj in objects:
        kind = catalog[obj['asset_id']]['kind']
        if kind != 'mug':
            if obj['support'] != 'floor' or not 0 <= obj['z'] <= .02:
                raise ValueError('Furniture must be on the floor, with at most 2 cm clearance')
        else:
            if obj['support'] != table['name']:
                raise ValueError('Mug must be supported by the table')
            top = table['z'] + catalog[table['asset_id']]['dimensions']['height']
            if not -.00001 <= obj['z'] - top <= .02:
                raise ValueError(f'Mug bottom z must be between {top:.6f} and {top+.02:.6f}')
            mb = footprint(obj, catalog)
            if not (mb[0] >= tb[0]+.05 and mb[1] >= tb[1]+.05 and mb[2] <= tb[2]-.05 and mb[3] <= tb[3]-.05):
                raise ValueError('Mug must be at least 5 cm inside the table footprint')
    furniture = [o for o in objects if o['support']=='floor']
    for i, a in enumerate(furniture):
        ab = footprint(a, catalog)
        for b in furniture[i+1:]:
            bb = footprint(b, catalog)
            if min(ab[2],bb[2])-max(ab[0],bb[0]) > -.05 and min(ab[3],bb[3])-max(ab[1],bb[1]) > -.05:
                raise ValueError(f'{a["name"]} and {b["name"]} overlap or have less than 5 cm clearance')
    return copy.deepcopy(plan)


def package_scene(plan, catalog, fixture, scene_dir, prompt, scene_id):
    plan = validate_plan(plan, catalog)
    if not re.fullmatch(r'[a-f0-9]{8,32}', scene_id):
        raise ValueError('Invalid scene ID')
    fixture, scene_dir = Path(fixture), Path(scene_dir)
    original = json.loads((fixture/'layout_84b703fb.json').read_text())
    original_room = original['rooms'][0]
    room_id = f'room_evoworld_{scene_id}'
    object_ids = {o['name']: f'{room_id}_{o["name"]}' for o in plan['objects']}
    width,length,height = plan['width'],plan['length'],plan['height']
    wall_material = original_room['walls'][0]['material']
    def point(x,y,z=0): return {'x':x,'y':y,'z':z}
    corners=[(0,0),(width,0),(width,length),(0,length)]
    walls=[{'id':f'wall_{room_id}_{i}','start_point':point(*a),'end_point':point(*corners[(i+1)%4]),'height':height,'thickness':.1,'material':wall_material} for i,a in enumerate(corners)]
    objects=[]
    for obj in plan['objects']:
        asset=catalog[obj['asset_id']]
        objects.append({'id':object_ids[obj['name']],'room_id':room_id,'type':asset['kind'],
            'description':asset['description'],'position':point(obj['x'],obj['y'],obj['z']),
            'rotation':point(0,0,obj['yaw']),'dimensions':asset['dimensions'],
            'source':'generation','source_id':obj['asset_id'],
            'place_id':'floor' if obj['support']=='floor' else object_ids[obj['support']],
            'place_guidance':'GPT-generated placement using an unchanged SAGE-10k asset',
            'placement_constraints':[],'mass':asset['mass'],'pbr_parameters':asset['pbr_parameters']})
    room={'id':room_id,'room_type':'small_meeting_workspace','position':point(0,0),'dimensions':{'width':width,'length':length,'height':height},'walls':walls,'doors':[],'windows':[],'objects':objects,'floor_material':original_room['floor_material'],'ceiling_height':height}
    layout={'id':f'layout_evoworld_{scene_id}','rooms':[room],'total_area':width*length,'building_style':'official asset materials reused','description':plan['title'],'created_from_text':prompt,'policy_analysis':None}
    scene_dir.mkdir(parents=True,exist_ok=False)
    (scene_dir/'objects').mkdir();(scene_dir/'materials').mkdir()
    copied=[]
    for asset_id in sorted({o['asset_id'] for o in plan['objects']}):
        for suffix in ['.ply','_texture.png']:
            src=fixture/'objects'/(asset_id+suffix);dst=scene_dir/'objects'/src.name
            shutil.copy2(src,dst);copied.append(dst)
    for material in [room['floor_material'],wall_material]:
        src=fixture/'materials'/(material+'.png');dst=scene_dir/'materials'/src.name
        shutil.copy2(src,dst);copied.append(dst)
    layout_path=scene_dir/'layout.json'
    layout_path.write_text(json.dumps(layout,indent=2))
    (scene_dir/'plan.json').write_text(json.dumps(plan,indent=2))
    provenance={'scene_id':scene_id,'new_layout':True,'new_mesh_generation':False,
        'dataset':'nvidia/SAGE-10k','dataset_revision':DATASET_REVISION,'sage_commit':SAGE_COMMIT,
        'source_layout_sha256':hashlib.sha256((fixture/'layout_84b703fb.json').read_bytes()).hexdigest(),
        'layout_sha256':hashlib.sha256(layout_path.read_bytes()).hexdigest(),
        'asset_sha256':{str(p.relative_to(scene_dir)):hashlib.sha256(p.read_bytes()).hexdigest() for p in copied}}
    (scene_dir/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return {'scene_id':scene_id,'layout_path':str(layout_path),'object_count':4,'new_layout':True,'new_mesh_generation':False}
