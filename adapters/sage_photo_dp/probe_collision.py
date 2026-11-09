"""Execute existing Phase7 simulator probe with the real original cuRobo planner."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'adapters/sage_realtosim'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
source=root/'adapters/sage_realtosim/runtime.py'
text=source.read_text().replace('out=Path(args.output);','out=Path(args.output).resolve();')
needle="    adapter.reset(seed=task['seed'])"
insert='''
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'adapters/sage_robot/src/server'))
    from collision_bridge import bridge_planner_class
    from isaaclab.curobo_tools.curobo_planner import MotionPlanner
    native_specs={o['id']:{'dynamic':o['dynamic']} for o in scene['objects']}
    Planner=bridge_planner_class(MotionPlanner,native_specs)
    planner=Planner(env,collision_checker=True,reference_prim_path='/World/envs/env_0/Robot',
        ignore_substring=['/World/envs/env_0/Robot','/World/GroundPlane','/World/Ground',
                          '/World/collisions','/World/light','/curobo'])
    import os
    (out/'isaaclab/test_obstacles').mkdir(parents=True,exist_ok=True)
    os.chdir(out)
    planner.add_obstacle()
    write('collision-coverage.json',{'objects':planner.native_collision_report,
         'keys':list(planner.obstacles_mesh),'complete':len(planner.native_collision_report)==6})
    q=robot.data.joint_pos[:,:7].clone()
    state=planner.kin_model.get_state(q)
    goal=state.ee_position.clone();goal[:,2]+=.03
    ee_plan,joint_plan=planner.plan_motion(q,goal,state.ee_quaternion.clone(),max_attempts=5)
    write('planning-report.json',{'success':joint_plan is not None,
        'target_offset_m':[0,0,.03],
        'waypoints':len(joint_plan.position) if joint_plan is not None else 0,
        'scope':'real cuRobo planning with all native collider meshes; existing Phase7 probe executes its own Cartesian motion'})
    if joint_plan is None:raise RuntimeError('cuRobo short-motion planning failed; see planning-report.json')
    if joint_plan is not None:
        np.savez_compressed(out/'planned-joints.npz',positions=cpu(joint_plan.position),
             velocities=cpu(joint_plan.velocity))
'''
assert text.count(needle)==1
text=text.replace(needle,needle+insert)
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
