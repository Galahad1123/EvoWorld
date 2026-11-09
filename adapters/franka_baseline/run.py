"""Bounded physical Franka task in a previously generated SAGE room."""
import argparse,json,os,sys,time,traceback,hashlib
from pathlib import Path
import numpy as np
from contract import assess_episode,validate_arrays,resolve_command
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[2];out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
scene=root/'experiments/phase2-small-20261107-200457-n70snxiw/central_4x4/sage-generated-20261107-200457-c3w6_1au/30dd907ee170/scene.usda'
asset=root/'assets/isaac6/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd'
started=time.monotonic();app=None

def save(name,data):
    (out/name).write_text(json.dumps(data,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))

try:
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':960,'height':720})
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('isaacsim.robot.manipulators.examples')
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.utils.types import ArticulationAction
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
    from pxr import UsdGeom,UsdLux,Gf,PhysxSchema
    import h5py
    assert omni.usd.get_context().open_stage(str(scene))
    for _ in range(10):app.update()
    stage=omni.usd.get_context().get_stage()
    world=World(physics_dt=1/60,rendering_dt=1/60,stage_units_in_meters=1.0,physics_prim_path='/World/PhysicsScene')
    world.get_physics_context().enable_fabric(False)
    world.get_physics_context().set_physx_update_transformations_settings(update_to_usd=True)
    base=np.array([1.45,1.97,.800]);initial=np.array([1.95,1.80,.825]);target=np.array([1.95,2.20,.825])
    robot=world.scene.add(Franka('/World/Franka',usd_path=str(asset),position=base,gripper_open_position=np.array([.04,.04])))
    home=np.array([0,-.569,0,-2.810,0,3.037,.741,.04,.04])
    robot.set_joints_default_state(positions=home,velocities=np.zeros(9))
    robot.gripper.set_default_state(joint_positions=home[-2:].copy())
    material=PhysicsMaterial('/World/CubeMaterial',static_friction=1.,dynamic_friction=1.,restitution=0.)
    cube=world.scene.add(DynamicCuboid('/World/TaskCube',name='task_cube',position=initial,size=.04,mass=.05,color=np.array([.8,.08,.06]),physics_material=material))
    cube.set_default_state(position=initial,orientation=np.array([1.,0,0,0]))
    body_api=PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath('/World/TaskCube'))
    body_api.CreateSolverPositionIterationCountAttr(32);body_api.CreateSolverVelocityIterationCountAttr(8)
    world.reset()
    event_steps=[180,240,30,60,120,180,240,60,120,60]
    events_dt=[1.0/x for x in event_steps]
    controller=PickPlaceController('baseline',robot.gripper,robot,end_effector_initial_height=1.10,events_dt=events_dt)
    assert robot.dof_names==['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7','panda_finger_joint1','panda_finger_joint2'],robot.dof_names
    # Reset contract: perturb the task body, then reset scene and controller.
    cube.set_world_pose(position=initial+np.array([0,0,.1]))
    world.reset();controller.reset(end_effector_initial_height=1.10)
    reset_error=float(np.linalg.norm(cube.get_world_pose()[0]-initial))
    save('reset.json',{'cube_error_m':reset_error,'actual_joints':robot.get_joint_positions(),'desired_joints':home})
    assert reset_error<1e-5 and np.max(abs(robot.get_joint_positions()-home))<1e-5
    for _ in range(60):world.step(render=False)
    import omni.replicator.core as rep
    from PIL import Image
    light=UsdLux.RectLight.Define(stage,'/World/TaskLight');light.CreateIntensityAttr(3000);light.CreateWidthAttr(3);light.CreateHeightAttr(3)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(2,2,2.5))
    camera=rep.create.camera(position=(.65,.65,2.25),look_at=(1.95,2.,.9),focal_length=24)
    product=rep.create.render_product(camera,(960,720));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
    def capture(name):
        t=world.current_time
        for _ in range(5):world.render()
        pixels=rgb.get_data();assert pixels.shape[:2]==(720,960) and pixels[:,:,:3].std()>10
        assert world.current_time==t,'Rendering advanced physics'
        Image.fromarray(pixels).save(out/name)
    capture('initial.png')
    def observation():
        cp,cq=cube.get_world_pose();ep,eq=robot.end_effector.get_world_pose()
        return {'joint_pos':robot.get_joint_positions().copy(),'joint_vel':robot.get_joint_velocities().copy(),
                'object_pos':cp.copy(),'object_quat':cq.copy(),'object_vel':cube.get_linear_velocity().copy(),
                'ee_pos':ep.copy(),'ee_quat':eq.copy(),'target_pos':target.copy()}
    states=[observation()];actions=[];events=[];callback_dts=[]
    world.add_physics_callback('phase3_clock',lambda dt:callback_dts.append(float(dt)))
    start_step=world.current_time_step_index;start_time=world.current_time
    command=states[0]['joint_pos'].copy();done_step=None
    for i in range(1800):
        event=int(controller.get_current_event());events.append(event)
        action=controller.forward(picking_position=states[-1]['object_pos'],placing_position=target,current_joint_positions=states[-1]['joint_pos'])
        dense=resolve_command(command,action.joint_positions,action.joint_velocities,action.joint_indices)
        command=dense[:9].copy()
        assert np.isfinite(dense).all()
        robot.apply_action(ArticulationAction(joint_positions=command,joint_velocities=dense[9:].copy()))
        actions.append(dense.copy());world.step(render=False);states.append(observation())
        if controller.is_done() and done_step is None:done_step=i
        if done_step is not None and i-done_step>=120:break
    arrays={key:np.stack([s[key] for s in states]) for key in states[0]}
    actions=np.stack(actions);n=len(actions)
    validate_arrays(actions,arrays['joint_pos'],arrays['joint_pos'][1:])
    timing={'steps':n,'observed_steps':world.current_time_step_index-start_step,'callback_count':len(callback_dts),
            'dt':1/60,'simulation_seconds':world.current_time-start_time,'callback_seconds':sum(callback_dts),'startup_settle_steps':60}
    assert timing['observed_steps']==n==len(callback_dts) and abs(timing['simulation_seconds']-n/60)<1e-4
    metrics=assess_episode(arrays['object_pos'],arrays['object_vel'],arrays['joint_pos'][:,-2:],target)
    metrics['controller_done']=controller.is_done();metrics['success']=metrics['success'] and controller.is_done()
    metadata={'schema':'evoworld.franka_joint_pos_vel.v1','source_sage_commit':'fe3966013a495dc1642710de93a83ddf66e534a0',
        'scene':str(scene),'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),'robot_asset':str(asset),
        'robot_base_world_m':base,'cube_initial_world_m':initial,'target_world_m':target,'cube_size_m':.04,'cube_mass_kg':.05,
        'joint_names':robot.dof_names,'action_type':'18D: q[0:9] (7 rad, 2 m), qdot[9:18] (7 rad/s, 2 m/s)',
        'pose_frame':'world, meters, Z-up; quaternion wxyz','ee_link':'panda_rightfinger (not tool center)',
        'policy':'Isaac Sim 6 built-in RMPflow PickPlaceController; not original SAGE M2T2/cuRobo',
        'controller_event_steps':event_steps,'cube_solver_iterations':[32,8],
        'physics_dt':1/60,'reset_error_m':reset_error,'max_steps':1800,'metrics':metrics,'timing':timing}
    rewards=-np.linalg.norm(arrays['object_pos'][1:]-target,axis=1)
    dones=np.zeros(n,dtype=np.uint8);dones[-1]=1
    with h5py.File(out/'trajectory.hdf5','w') as f:
        data=f.create_group('data');data.attrs['total']=n;data.attrs['env_args']=json.dumps(metadata,default=lambda x:x.tolist())
        demo=data.create_group('demo_0');demo.attrs['num_samples']=n;demo.attrs['success']=metrics['success']
        demo.create_dataset('actions',data=actions);demo.create_dataset('rewards',data=rewards);demo.create_dataset('dones',data=dones)
        demo.create_dataset('terminated',data=dones if metrics['success'] else np.zeros(n,dtype=np.uint8))
        demo.create_dataset('truncated',data=np.zeros(n,dtype=np.uint8) if metrics['success'] else dones)
        demo.create_dataset('controller_event',data=np.array(events));demo.create_dataset('time',data=np.arange(n)/60)
        for key,a in arrays.items():
            demo.create_dataset('obs/'+key,data=a[:-1]);demo.create_dataset('next_obs/'+key,data=a[1:])
    save('task.json',metadata)
    capture('final.png')
    # Absolute source references retained: this stage is an on-host diagnostic, not a portable package.
    stage.GetRootLayer().Export(str(out/'task-scene.usda'))
    save('report.json',{'passed':metrics['success'],'metrics':metrics,'timing':timing,'reset_error_m':reset_error,
                        'wall_seconds':time.monotonic()-started,'dataset':'trajectory.hdf5','new_gpt_calls':0})
    print('EVOWORLD_FRANKA_BASELINE_COMPLETE '+json.dumps(metrics),flush=True)
except BaseException:
    save('error.json',{'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-started})
    traceback.print_exc()
finally:
    if app is not None:app.close()
