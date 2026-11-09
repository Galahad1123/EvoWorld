"""Actual bounded asset physics and frozen-state previews; no robot/control labels."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

run = Path(sys.argv[1]).resolve()
out = run / 'physics'
out.mkdir(exist_ok=False)
started = time.monotonic()
app = None
exit_status = 1
try:
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'width': 800, 'height': 600})
    import omni.usd
    import omni.replicator.core as rep
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Gf
    from PIL import Image
    import h5py
    stage_path = run / 'relocated/scene-package/scene.usda'
    assert omni.usd.get_context().open_stage(str(stage_path))
    for _ in range(5):
        app.update()
    stage = omni.usd.get_context().get_stage()
    scene = UsdPhysics.Scene.Get(stage, '/World/PhysicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0., 0., -1.))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1. and UsdGeom.GetStageUpAxis(stage) == 'Z'
    spec = json.loads((run / 'scene.json').read_text())
    block = next(x for x in spec['objects'] if x['id'] == 'block')
    initial = np.asarray(block['position_m']) + [0, 0, .04]
    # Author the initial pose before PhysX initialization so USD and physics agree.
    # A post-reset set_world_pose updates PhysX only while the simulation is active.
    stage.GetPrimAtPath('/World/block').GetAttribute('xformOp:translate').Set(Gf.Vec3d(*initial))
    world = World(physics_dt=1 / 60, rendering_dt=1 / 60, stage_units_in_meters=1., physics_prim_path='/World/PhysicsScene')
    world.get_physics_context().enable_fabric(False)
    world.get_physics_context().set_physx_update_transformations_settings(update_to_usd=True)
    if len(sys.argv) > 2:
        # Retry only preview generation from a retained, already integrated physics trace.
        source = Path(sys.argv[2]).resolve()
        with np.load(source / 'integrated-states.npz') as saved:
            timestamps, poses, velocities, angular = [saved[k].copy() for k in ['time', 'poses', 'velocities', 'angular']]
        integration = json.loads((source / 'integration.json').read_text())
        timing, measured_mass = integration['timing'], integration['mass_kg']
        (out / 'reused-integration-source.json').write_text(json.dumps({'source': str(source), 'reason': 'retry frozen preview only; no new physics rollout'}, indent=2))
    else:
        body = world.scene.add(SingleRigidPrim('/World/block', name='block'))
        world.reset()
        assert np.allclose(body.get_world_pose()[0], initial, atol=1e-6)
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(np.zeros(3))
        callbacks = []
        world.add_physics_callback('phase5_clock', lambda dt: callbacks.append(float(dt)))
        start_time, start_step = world.current_time, world.current_time_step_index
    
        def state():
            pos, quat = body.get_world_pose()
            return np.r_[pos, quat], body.get_linear_velocity().copy(), body.get_angular_velocity().copy()
    
        poses, velocities, angular = [], [], []
        timestamps = [0.]
        p, v, w = state()
        poses.append(p); velocities.append(v); angular.append(w)
        for _ in range(120):
            world.step(render=False)
            p, v, w = state()
            poses.append(p); velocities.append(v); angular.append(w)
            timestamps.append(world.current_time - start_time)
        poses, velocities, angular, timestamps = map(np.asarray, [poses, velocities, angular, timestamps])
        timing = {'steps': world.current_time_step_index - start_step, 'callback_count': len(callbacks),
                  'simulation_seconds': world.current_time - start_time, 'callback_seconds': sum(callbacks), 'dt': 1 / 60}
        assert timing['steps'] == timing['callback_count'] == 120
        assert abs(timing['simulation_seconds'] - 2.) < 1e-5 and abs(timing['callback_seconds'] - 2.) < 1e-5
        measured_mass = float(body.get_mass())
        world.remove_physics_callback('phase5_clock')
    np.savez_compressed(out / 'integrated-states.npz', time=timestamps, poses=poses, velocities=velocities, angular=angular)
    (out / 'integration.json').write_text(json.dumps({'timing': timing, 'mass_kg': measured_mass}, indent=2))
    world.stop()
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath('/World/block')).CreateRigidBodyEnabledAttr(False)
    light = UsdLux.RectLight.Define(stage, '/World/InspectionLight')
    light.CreateIntensityAttr(1500.)
    light.CreateWidthAttr(2.)
    light.CreateHeightAttr(2.)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(.3, -.3, 2.5))
    views = {}
    for name, pos in [('front', (1.7, -1.6, 1.7)), ('back', (-1.7, 1.6, 1.7))]:
        cam = rep.create.camera(position=pos, look_at=(0, 0, .65), focal_length=24.)
        product = rep.create.render_product(cam, (800, 600))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach(product)
        views[name] = (cam, rgb, pos)
    images, camera_meta = [], []

    def capture(name, view, state_index, lighting):
        before = world.current_time
        assert not world.is_playing()
        cam, rgb, pos = views[view]
        physics_pos, physics_quat = poses[state_index, :3], poses[state_index, 3:]
        # Offline preview of a recorded state, with dynamics stopped and rigid body disabled.
        block_prim = stage.GetPrimAtPath('/World/block')
        prior_displayed_position = list(UsdGeom.Xformable(block_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
        block_prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*map(float, physics_pos)))
        block_prim.GetAttribute('xformOp:orient').Set(Gf.Quatf(float(physics_quat[0]), Gf.Vec3f(*map(float, physics_quat[1:]))))
        rep.orchestrator.step(rt_subframes=5, delta_time=0.0, pause_timeline=True, wait_for_render=True)
        # Check the displayed transform against the integrated state, not just the clock.
        body_matrix = UsdGeom.Xformable(stage.GetPrimAtPath('/World/block')).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        displayed_pos = np.asarray(body_matrix.ExtractTranslation())
        displayed_q = body_matrix.ExtractRotationQuat()
        displayed_quat = np.r_[displayed_q.GetReal(), displayed_q.GetImaginary()]
        assert np.allclose(displayed_pos, physics_pos, atol=1e-5), (displayed_pos, physics_pos)
        assert abs(np.dot(displayed_quat, physics_quat)) > 1 - 1e-5
        pixels = rgb.get_data()[:, :, :3].copy()
        assert pixels.shape == (600, 800, 3) and pixels.std() > 10
        assert world.current_time == before, 'Image render advanced physics'
        Image.fromarray(pixels).save(out / (name + '.png'))
        images.append(pixels)
        prims = cam.get_output_prims()['prims']
        # Replicator may return an Xform containing the Camera, not the Camera itself.
        cameras = [p for root in prims for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Camera)]
        assert len(cameras) == 1, f'Expected one camera under Replicator outputs, got {len(cameras)}'
        camera_prim = cameras[0]
        c = UsdGeom.Camera(camera_prim)
        camera_meta.append({'name': name, 'view': view, 'state_index': state_index, 'lighting': lighting,
            'displayed_block_pose_wxyz': np.r_[displayed_pos, displayed_quat].tolist(),
            'prior_displayed_position_m': prior_displayed_position,
            'pose_sync': 'offline frozen-state rerender from previously integrated pose; physics stopped, not live sensor data',
            'position_m': pos, 'look_at_m': [0, 0, .65],
            'world_T_camera_usd_column_convention': np.asarray(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T.tolist(),
            'focal_length_mm': c.GetFocalLengthAttr().Get(), 'horizontal_aperture_mm': c.GetHorizontalApertureAttr().Get(),
            'vertical_aperture_mm': c.GetVerticalApertureAttr().Get(), 'resolution': [800, 600],
            'calibration_status': 'authored USD camera metadata, not independently calibrated sensor'})

    capture('initial-front', 'front', 0, 'dome800+rect1500')
    capture('final-front', 'front', 120, 'dome800+rect1500')
    capture('final-back', 'back', 120, 'dome800+rect1500')
    light.GetIntensityAttr().Set(300.)
    capture('final-front-low-light', 'front', 120, 'dome800+rect300')
    with h5py.File(out / 'asset-physics.hdf5', 'w') as f:
        f.attrs.update(schema='evoworld.asset_physics.v1', units='m,s,rad', up_axis='Z', quaternion='wxyz',
            state_semantics='actual integrated rigid-body states; not actions or policy demonstration',
            image_semantics='offline frozen-state rerenders after physics stopped; state_index associates recorded pose, not live sensor timestamp',
            cameras=json.dumps(camera_meta), timing=json.dumps(timing))
        f['time'] = timestamps
        f['block/pose_world'] = poses
        f['block/linear_velocity_world'] = velocities
        f['block/angular_velocity_world'] = angular
        f.create_dataset('images/rgb', data=np.stack(images), compression='gzip', compression_opts=1)
        f['images/state_index'] = np.array([0, 120, 120, 120])
    (out / 'camera-metadata.json').write_text(json.dumps(camera_meta, indent=2))
    (out / 'execution.json').write_text(json.dumps({'completed': True, 'timing': timing,
        'wall_seconds': time.monotonic() - started, 'mass_kg': measured_mass,
        'initial_position_m': initial.tolist(), 'drop_offset_m': .04,
        'note': 'Only this validation process raises the block 4 cm. Portable scene retains supported placement.'}, indent=2))
    print('PHASE5_ASSET_PHYSICS_COMPLETE', flush=True)
    exit_status = 0
except BaseException:
    (out / 'error.json').write_text(json.dumps({'traceback': traceback.format_exc()}, indent=2))
    traceback.print_exc()
finally:
    if app is not None:
        # fast_shutdown may terminate the interpreter: pass the actual result explicitly.
        app.close(exit_code=exit_status)
    raise SystemExit(exit_status)
