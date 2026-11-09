"""Minimal MCP surface over official SAGE asset and Isaac import utilities."""
import contextlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from scene_contract import catalog_from_fixture,package_scene
from acceptance import checked_report,SUBPROCESS_TIMEOUT_SECONDS,read_run_config

ROOT=Path(__file__).resolve().parents[2]
FIXTURE=ROOT/'assets/SAGE-10k/extracted'
CATALOG=catalog_from_fixture(FIXTURE)
RUN=Path(os.environ['EVOWORLD_RUN_DIR']).resolve()
if not RUN.is_relative_to(ROOT/'experiments') or not RUN.is_dir():
    raise RuntimeError('Run directory must be an existing EvoWorld experiment directory')
CONFIG=read_run_config(RUN)
PROMPT=(RUN/'prompt.txt').read_text()
scenes={}
simulation_count=0
mcp=FastMCP('EvoWorld-minimal-SAGE')

@mcp.tool()
def list_assets() -> dict:
    """List fixed-dimension official assets; select one table, two chairs and one mug."""
    return {'assets':list(CATALOG.values()),'units':'meters; z up; XY position is mesh center, Z is mesh bottom',
        'build_schema':{'title':'short title','width':'3..6','length':'3..6','height':'2.4..3.2',
        'objects':[{'name':'unique lowercase identifier','asset_id':'from catalog','x':'number','y':'number',
                    'z':'bottom height','yaw':'0|90|180|270','support':'floor or table object name'}]},
        'constraints':['Exactly four objects: table, chair, chair, mug. A chair asset may be reused.',
            'Furniture bottom z between 0 and 0.02 m. At least 0.08 m from room boundaries and 0.05 m between furniture boxes.',
            'Mug supported by table; bottom z = table.z + table height + 0.005. Keep 0.05 m inside table footprint.',
            'Prefer a simple central table with chairs on opposite sides and mug near tabletop center. No doors/windows in this bounded test.'],
        'scope':'new layout from reused SAGE-10k mesh/material assets; not TRELLIS/MatFuse generation'}

@mcp.tool()
def build_scene(plan: dict) -> dict:
    """Build a fresh SAGE FloorPlan JSON/GLB from the exact plan schema returned by list_assets."""
    if len(scenes)>=4:
        raise ValueError('Build limit reached')
    scene_id=uuid.uuid4().hex[:12]
    scene_dir=RUN/scene_id
    result=package_scene(plan,CATALOG,FIXTURE,scene_dir,PROMPT,scene_id)
    sys.path.insert(0,str(ROOT/'assets/SAGE-10k/kits'))
    with (scene_dir/'export.log').open('w') as log,contextlib.redirect_stdout(log):
        from export_glb import export_glb
        export_glb(result['layout_path'],str(scene_dir))
    import trimesh
    import numpy as np
    geometry=trimesh.load(scene_dir/'layout.glb',force='scene')
    if len(geometry.geometry)!=10 or not all(np.isfinite(m.vertices).all() for m in geometry.geometry.values()):
        raise ValueError('Official SAGE GLB export did not produce ten finite meshes')
    (scene_dir/'geometry.json').write_text(json.dumps({'mesh_count':10,'finite_geometry':True,'bytes':(scene_dir/'layout.glb').stat().st_size},indent=2))
    scenes[scene_id]={'directory':scene_dir,'simulation_passed':False}
    return {**result,'mesh_count':10,'next_step':'Call simulate_scene with scene_id; review its rendered image before finish_scene.'}

def _scene(scene_id):
    if scene_id not in scenes:
        raise ValueError('Unknown scene ID; only scenes built in this run are accepted')
    return scenes[scene_id]

def _run(scene_dir,render=False):
    command=[str(ROOT/'envs/isaacsim6/bin/python'),'-u',str(Path(__file__).with_name('simulate_scene.py')),'--scene',str(scene_dir),'--steps',str(CONFIG['physics_steps'])]
    log_name='render.log' if render else 'physics.log'
    if render: command.append('--render-only')
    with (scene_dir/log_name).open('w') as log:
        completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=SUBPROCESS_TIMEOUT_SECONDS)
    return checked_report(scene_dir,completed.returncode,log_name,'render.json' if render else 'physics.json',
        'EVOWORLD_GENERATED_RELOAD_PASS' if render else 'EVOWORLD_GENERATED_PHYSICS_EXECUTED')

@mcp.tool()
def simulate_scene(scene_id: str) -> dict:
    """Run real Isaac physics, then independently reload USD and render. Failed physics needs a new layout."""
    global simulation_count
    state=_scene(scene_id)
    if state.get('attempted'):
        raise ValueError('This scene already has an immutable simulation attempt; build a new scene to revise it')
    if simulation_count>=CONFIG['max_simulations']:
        raise ValueError('Simulation budget exhausted')
    simulation_count+=1;state['attempted']=True
    physics=_run(state['directory'])
    render=_run(state['directory'],render=True)
    state['simulation_passed']=physics['passed'] and render['passed']
    return {'scene_id':scene_id,'passed':state['simulation_passed'],
        'body_count':physics['body_count'],'timing':physics['timing'],'whole_trajectory':physics['whole_trajectory'],'diagnostics':physics['diagnostics'],
        'floor_support_ok':physics['floor_support_ok'],'mug_support_ok':physics['mug_support_ok'],
        'mug_table_vertical_gap_m':physics['mug_table_vertical_gap_m'],
        'preview_path':str(state['directory']/'preview.png'),'rendered_state':render['rendered_state'],
        'motion_check':physics['motion_check'],
        'next_step':'Inspect the actual supplied render. If acceptable call finish_scene; otherwise build a revised plan.'}

@mcp.tool()
def finish_scene(scene_id: str,visual_ok: bool,visual_notes: str) -> dict:
    """Accept only a physics/reload-passing scene after reviewing the rendered image."""
    state=_scene(scene_id)
    if not state['simulation_passed']:
        raise ValueError('Cannot finish before successful physics and independent reload')
    if not visual_ok or len(visual_notes.strip())<10:
        raise ValueError('Visual review must accept the scene with substantive observations')
    report={'passed':True,'scene_id':scene_id,'visual_ok':visual_ok,'visual_notes':visual_notes,
        'scope':'GPT-generated layout, reused official assets, real MCP, SAGE conversion and Isaac validation',
        'new_mesh_generation':False,'scene_directory':str(state['directory'])}
    (RUN/'completion.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    mcp.run(transport='stdio')
