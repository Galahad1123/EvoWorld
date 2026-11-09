"""Import an actual generated SAGE layout; record physics or independently reload/render."""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--scene',required=True,type=Path)
parser.add_argument('--render-only',action='store_true')
parser.add_argument('--steps',type=int,choices=[30,120,600],default=120)
args=parser.parse_args()
scene_dir=args.scene.resolve()
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'assets/SAGE-10k/kits'))
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
import isaacsim_utils as official
import numpy as np
import trimesh
from acceptance import assess_trace_motion,trajectory_limits
from fixed_step import simulate_fixed_steps
layout=json.loads((scene_dir/'layout.json').read_text())
room=layout['rooms'][0]
objects=room['objects']
expected_ids={o['id'] for o in objects}

def encode(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)

def save(name,data):
    (scene_dir/name).write_text(json.dumps(data,default=encode,indent=2))

try:
    official.start_simulation_app()
    import omni.usd
    from pxr import Usd,UsdGeom,UsdPhysics,UsdLux,Sdf,Gf,PhysxSchema
    if args.render_only:
        assert omni.usd.get_context().open_stage(str(scene_dir/'scene.usda'))
        for _ in range(20): official.simulation_app.update()
        stage=omni.usd.get_context().get_stage()
    else:
        track_ids=official.get_layout_scene_loaded(str(scene_dir/'layout.json'))
        assert set(track_ids)==expected_ids,(track_ids,expected_ids)
        stage=omni.usd.get_context().get_stage()
    assert UsdGeom.GetStageMetersPerUnit(stage)==1.0
    assert UsdGeom.GetStageUpAxis(stage)==UsdGeom.Tokens.z
    bodies=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert {p.GetName() for p in bodies}==expected_ids
    # Package texture paths relative to the USD root for independent relocation/reload.
    texture_files=[]
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName()!=Sdf.ValueTypeNames.Asset: continue
            asset=attr.Get()
            if not asset or not asset.path: continue
            path=Path(asset.path)
            resolved=path if path.is_absolute() else scene_dir/path
            assert resolved.is_file(),f'Missing texture: {resolved}'
            assert resolved.resolve().is_relative_to(scene_dir)
            texture_files.append(str(resolved.relative_to(scene_dir)))
            if not args.render_only: attr.Set(Sdf.AssetPath(os.path.relpath(resolved,scene_dir)))
    assert len(texture_files)>=4
    if args.render_only:
        import omni.replicator.core as rep
        from PIL import Image
        width,length,height=(room['dimensions'][k] for k in ['width','length','height'])
        light=UsdLux.RectLight.Define(stage,'/World/InspectionCeilingLight')
        light.CreateIntensityAttr(3000)
        light.CreateWidthAttr(width*.8);light.CreateHeightAttr(length*.8)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(width/2,length/2,height-.1))
        position=(.3,.3,min(height-.3,2.4));target=(width/2,length/2,.65)
        camera=rep.create.camera(position=position,look_at=target,focal_length=16)
        product=rep.create.render_product(camera,(960,720))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
        rep.orchestrator.step(rt_subframes=8,pause_timeline=True)
        pixels=rgb.get_data()
        assert pixels.shape[:2]==(720,960) and np.isfinite(pixels).all()
        assert pixels[:,:,:3].std()>10,'Blank or nearly uniform rendering'
        Image.fromarray(pixels).save(scene_dir/'preview.png')
        save('render.json',{'passed':True,'body_count':len(bodies),'texture_files':sorted(set(texture_files)),
            'camera_position':position,'camera_target':target,'image_shape':pixels.shape,
            'pixel_std':float(pixels[:,:,:3].std()),'inspection_light_added':True,
            'rendered_state':'initial layout reloaded from saved USD; physics trajectories stored separately'})
        print('EVOWORLD_GENERATED_RELOAD_PASS',flush=True)
    else:
        physics=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene')
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));physics.CreateGravityMagnitudeAttr(9.81)
        PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim()).CreateTimeStepsPerSecondAttr(60)
        points={p.GetName():np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float) for p in bodies}
        stage.GetRootLayer().Export(str(scene_dir/'scene.usda'))
        result=simulate_fixed_steps(official,track_ids,args.steps)
        assert result['status']=='success'
        stats=official.generate_physics_statistics(result['traced_data_all'],track_ids)
        diagnostics={};final_bounds={};whole_trajectory={}
        for obj in objects:
            name=obj['id'];trace=result['traced_data_all']['/World/'+name]
            pos=np.asarray(trace['final_position']);quat=np.asarray(trace['final_orientation'])
            trajectory=np.asarray(trace['position_traj'])
            assert trajectory.shape==(args.steps+1,3) and np.isfinite(trajectory).all()
            assert np.isfinite(pos).all() and np.isfinite(quat).all()
            rotation=trimesh.transformations.quaternion_matrix(quat)[:3,:3]
            moved=points[name]@rotation.T+pos
            before=np.array([points[name].min(axis=0),points[name].max(axis=0)])
            after=np.array([moved.min(axis=0),moved.max(axis=0)])
            final_bounds[name]=after
            whole_trajectory[name]=trajectory_limits(trace,before.mean(axis=0))
            center_shift=float(np.linalg.norm(after.mean(axis=0)-before.mean(axis=0)))
            angle=float(stats['objects'][name]['orientation_angle_offset'])
            diagnostics[name]={'center_displacement_m':center_shift,'rotation_change_degrees':angle,
                'final_bounds':after,'upstream_stable':bool(trace['stable']),
                'within_displacement_limit':center_shift<=.05,'within_rotation_limit':angle<=8.0}
        table=next(o for o in objects if o['type']=='table');mug=next(o for o in objects if o['type']=='mug')
        tb=final_bounds[table['id']];mb=final_bounds[mug['id']]
        mug_gap=float(mb[0,2]-tb[1,2])
        mug_supported=abs(mug_gap)<=.03 and all(mb[0,i]>=tb[0,i]-.01 and mb[1,i]<=tb[1,i]+.01 for i in [0,1])
        floor_ok=all(abs(float(final_bounds[o['id']][0,2]))<=.03 for o in objects if o['place_id']=='floor')
        all_stable=all(d['upstream_stable'] and d['within_displacement_limit'] and d['within_rotation_limit'] for d in diagnostics.values())
        motion_check=assess_trace_motion(result['traced_data_all'],expected_samples=args.steps+1,tail_samples=61 if args.steps>=600 else 10,reference_centers={'/World/'+name:(pts.min(axis=0)+pts.max(axis=0))/2 for name,pts in points.items()})
        report={'passed':bool(all_stable and floor_ok and mug_supported and motion_check['motion_observed'] and motion_check['tail_motion_bounded'] and all(x['passed'] for x in whole_trajectory.values())),'body_count':len(bodies),'meters_per_unit':1.0,
            'simulation_steps':args.steps,'timing':result['timing'],'whole_trajectory':whole_trajectory,'gravity_m_s2':9.81,'motion_check':motion_check,'diagnostics':diagnostics,'upstream_stats':stats,
            'floor_support_ok':bool(floor_ok),'mug_support_ok':bool(mug_supported),'mug_table_vertical_gap_m':mug_gap,
            'acceptance_limits':{'center_displacement_m':.05,'rotation_change_degrees':8,'support_tolerance_m':.03},
            'result':result}
        save('physics.json',report)
        print('EVOWORLD_GENERATED_PHYSICS_EXECUTED',json.dumps({'passed':report['passed'],'mug_gap':mug_gap,'stable':stats['stable_objects']}),flush=True)
except BaseException as exc:
    traceback.print_exc()
    save('render-error.json' if args.render_only else 'physics-error.json',{'error_type':type(exc).__name__,'traceback':traceback.format_exc()})
    raise
finally:
    if official.simulation_app is not None: official.simulation_app.close()
