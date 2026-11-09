"""Explicit, measured physics stepping; no app-update or manual timeline increments."""
import time
import numpy as np
from acceptance import validate_timing

def simulate_fixed_steps(official,track_ids,steps):
    from isaacsim.core.api import SimulationContext
    from pxr import UsdGeom,Usd
    prims,paths=official.get_all_prims_with_paths(track_ids)
    def sample():
        values=[]
        for prim in prims:
            transform=UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pos,quat=official.extract_position_orientation(transform)
            values.append((np.array(pos,dtype=float),np.array([quat.GetReal(),*quat.GetImaginary()],dtype=float)))
        return values
    authored=sample()
    context=SimulationContext(physics_dt=1/60,rendering_dt=1/60,stage_units_in_meters=1.,physics_prim_path='/World/PhysicsScene')
    physics=context.get_physics_context()
    physics.enable_fabric(False)
    physics.set_physx_update_transformations_settings(update_to_usd=True)
    context.initialize_physics()
    if not context.is_playing(): context.play()
    callback_dts=[]
    context.add_physics_callback('evoworld_measured_dt',lambda dt:callback_dts.append(float(dt)))
    start_step=context.current_time_step_index;start_time=context.current_time
    samples=[sample()]
    wall_start=time.monotonic()
    try:
        for _ in range(steps):
            context.step(render=False)
            samples.append(sample())
        timing=validate_timing(steps,context.current_time_step_index-start_step,context.current_time-start_time,callback_dts)
        timing.update(wall_seconds=time.monotonic()-wall_start,startup_steps_before_window=start_step,
                      startup_seconds_before_window=start_time,pose_samples=len(samples),mode='SimulationContext.step(render=False), USD transforms synchronized')
        traces={}
        for index,path in enumerate(paths):
            positions=np.array([frame[index][0] for frame in samples])
            orientations=np.array([frame[index][1] for frame in samples])
            initial_pos,initial_quat=authored[index]
            stable=np.linalg.norm(positions[-1]-initial_pos)<=.2 and official.quaternion_angle(initial_quat,orientations[-1])<=8.
            traces[path]={'initial_position':initial_pos,'initial_orientation':initial_quat,
                'final_position':positions[-1],'final_orientation':orientations[-1],
                'position_traj':positions,'orientation_traj':orientations,'stable':bool(stable)}
        return {'status':'success','traced_data_all':traces,'timing':timing}
    finally:
        context.remove_physics_callback('evoworld_measured_dt')
        context.stop()
