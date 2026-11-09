"""Single bounded real PhysX drop for every foreground rigid proxy. No replay images/actions."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

candidate=Path(sys.argv[1]).resolve();out=candidate/'physics';out.mkdir(exist_ok=False)
app=None;status=1;started=time.monotonic()
try:
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True})
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import UsdGeom, UsdPhysics, Gf
    import h5py
    assert omni.usd.get_context().open_stage(str(candidate/'relocated/scene.usda'))
    stage=omni.usd.get_context().get_stage()
    ids=[str(p.GetPath()).split('/')[-1] for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert set(ids)=={'milk','tissue'},ids
    initial={}
    for name in ids:
        p=stage.GetPrimAtPath('/World/'+name)
        attr=p.GetAttribute('xformOp:translate');xyz=np.array(attr.Get(),float);xyz[2]+=.02
        attr.Set(Gf.Vec3d(*xyz));initial[name]=xyz.tolist()
    scene=UsdPhysics.Scene.Get(stage,'/World/PhysicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
    world=World(physics_dt=1/60,rendering_dt=1/60,stage_units_in_meters=1.,physics_prim_path='/World/PhysicsScene')
    world.get_physics_context().enable_fabric(False)
    world.get_physics_context().set_physx_update_transformations_settings(update_to_usd=True)
    bodies={n:world.scene.add(SingleRigidPrim('/World/'+n,name=n)) for n in ids}
    world.reset()
    for n,b in bodies.items():
        assert np.allclose(b.get_world_pose()[0],initial[n],atol=1e-5)
        b.set_linear_velocity(np.zeros(3));b.set_angular_velocity(np.zeros(3))
    callbacks=[];world.add_physics_callback('photo_clock',lambda dt:callbacks.append(float(dt)))
    start_t,start_step=world.current_time,world.current_time_step_index
    times=[];states={n:[] for n in ids};velocities={n:[] for n in ids};angular={n:[] for n in ids}
    for i in range(121):
        times.append(world.current_time-start_t)
        for n,b in bodies.items():
            pos,quat=b.get_world_pose();states[n].append(np.r_[pos,quat])
            velocities[n].append(b.get_linear_velocity().copy());angular[n].append(b.get_angular_velocity().copy())
        if i<120:world.step(render=False)
    timing={'steps':world.current_time_step_index-start_step,'callback_count':len(callbacks),'seconds':world.current_time-start_t,'callback_seconds':sum(callbacks),'dt':1/60}
    assert timing['steps']==timing['callback_count']==120 and abs(timing['seconds']-2)<1e-5
    with h5py.File(out/'states.hdf5','w') as f:
        f.attrs.update(schema='evoworld.photo_asset_physics.v1',units='m,s,rad',up_axis='Z',quaternion='wxyz',
            semantics='actual full-rate PhysX rigid-proxy states; no robot actions, no cloth/liquid simulation, no synchronized images',timing=json.dumps(timing))
        f['time']=np.asarray(times)
        for n in ids:
            f[n+'/pose_world']=np.asarray(states[n]);f[n+'/linear_velocity_world']=np.asarray(velocities[n]);f[n+'/angular_velocity_world']=np.asarray(angular[n])
    (out/'execution.json').write_text(json.dumps({'completed':True,'timing':timing,'dynamic_ids':ids,'initial_positions':initial,
        'validation_drop_offset_m':.02,'actual_mass_kg':{n:float(b.get_mass()) for n,b in bodies.items()},'wall_seconds':time.monotonic()-started},indent=2))
    print('PHOTO_ASSET_PHYSICS_COMPLETE',flush=True);status=0
except BaseException:
    (out/'error.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2));traceback.print_exc()
finally:
    if app is not None:app.close(exit_code=status)
    raise SystemExit(status)
