import copy
import json
from pathlib import Path
import pytest
from scene_contract import validate_plan, catalog_from_fixture, package_scene

CATALOG = {
    'table': {'kind':'table', 'dimensions':{'width':1.4,'length':0.8,'height':0.8}},
    'chair': {'kind':'chair', 'dimensions':{'width':0.8,'length':0.8,'height':1.0}},
    'mug': {'kind':'mug', 'dimensions':{'width':0.12,'length':0.1,'height':0.1}},
}

def valid_plan():
    return {'title':'Small meeting space','width':4.0,'length':4.0,'height':2.7,'objects':[
        {'name':'table','asset_id':'table','x':2.,'y':2.,'z':0.005,'yaw':0,'support':'floor'},
        {'name':'chair_a','asset_id':'chair','x':2.,'y':0.9,'z':0.005,'yaw':0,'support':'floor'},
        {'name':'chair_b','asset_id':'chair','x':2.,'y':3.1,'z':0.005,'yaw':180,'support':'floor'},
        {'name':'mug','asset_id':'mug','x':2.,'y':2.,'z':0.81,'yaw':0,'support':'table'},
    ]}

def test_valid_support_and_separated_objects():
    p=valid_plan()
    assert validate_plan(p,CATALOG)==p

@pytest.mark.parametrize('change', ['unknown','outside','overlap','unsupported','duplicate','nan','wrong_count','wrong_height'])
def test_rejects_invalid_geometry_before_artifact_creation(change):
    p=valid_plan()
    if change=='unknown': p['objects'][0]['asset_id']='missing'
    if change=='outside': p['objects'][1]['x']=0.1
    if change=='overlap': p['objects'][1]['x']=2.;p['objects'][1]['y']=2.
    if change=='unsupported': p['objects'][3]['support']='chair_a'
    if change=='duplicate': p['objects'][3]['name']='table'
    if change=='nan': p['objects'][0]['x']=float('nan')
    if change=='wrong_count': p['objects'].pop()
    if change=='wrong_height': p['objects'][3]['z']=1.3
    with pytest.raises(ValueError): validate_plan(p,CATALOG)

def test_rotated_bounds_are_checked():
    p=valid_plan(); p['objects'][0].update(x=.5,yaw=90)
    p['objects'][3].update(x=.5)
    # A 90-degree table extends 0.4 along X: valid; yaw=0 extends 0.7: out of bounds.
    assert validate_plan(p,CATALOG)
    p['objects'][0]['yaw']=0
    with pytest.raises(ValueError): validate_plan(p,CATALOG)

def test_package_has_fresh_ids_and_official_parser_loads(tmp_path):
    root=Path(__file__).resolve().parents[3]
    fixture=root/'assets/SAGE-10k/extracted'
    catalog=catalog_from_fixture(fixture)
    plan=valid_plan()
    keys={kind:next(k for k,v in catalog.items() if v['kind']==kind) for kind in ['table','chair','mug']}
    for obj in plan['objects']: obj['asset_id']=keys[CATALOG[obj['asset_id']]['kind']]
    plan['objects'][3]['z']=catalog[keys['table']]['dimensions']['height']+.01
    result=package_scene(plan,catalog,fixture,tmp_path/'scene','User text', 'aabbccdd')
    data=json.loads(Path(result['layout_path']).read_text())
    assert data['id']=='layout_evoworld_aabbccdd'
    assert len(data['rooms'][0]['objects'])==4
    assert data['created_from_text']=='User text'
    assert all(o['id'].startswith('room_evoworld_aabbccdd_') for o in data['rooms'][0]['objects'])
    import sys
    sys.path.insert(0,str(root/'assets/SAGE-10k/kits'))
    from tex_utils_local import dict_to_floor_plan,export_layout_to_mesh_dict_list_v2
    meshes=export_layout_to_mesh_dict_list_v2(dict_to_floor_plan(data),str(tmp_path/'scene'))
    assert len(meshes)==10  # four objects, four walls, floor, ceiling
    assert sum(not v['static'] for v in meshes.values())==4
    assert all(Path(v['texture']['texture_map_path']).is_file() for v in meshes.values())
    assert all(isinstance(v['texture'].get('pbr_parameters',{}),dict) for v in meshes.values())
    original=json.loads((fixture/'layout_84b703fb.json').read_text())
    source_pbr={o['source_id']:o.get('pbr_parameters') for o in original['rooms'][0]['objects']}
    assert all(o['pbr_parameters']==source_pbr[o['source_id']] for o in data['rooms'][0]['objects'])
    assert (tmp_path/'scene'/'provenance.json').is_file()
