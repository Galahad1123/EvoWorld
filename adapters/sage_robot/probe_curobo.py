"""Real cuRobo v1 Franka collision-aware motion planning on RTX5090."""
import json,tempfile,time,traceback
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2];out=Path(tempfile.mkdtemp(prefix='sage-original-curobo-',dir=root/'experiments'));print(out,flush=True)
t0=time.monotonic()
try:
 from curobo.types.base import TensorDeviceType
 from curobo.types.math import Pose
 from curobo.types.robot import JointState
 from curobo.wrap.reacher.motion_gen import MotionGen,MotionGenConfig,MotionGenPlanConfig
 world={'cuboid':{'table':{'dims':[2.,2.,.1],'pose':[0.,0.,-.15,1.,0.,0.,0.]}}}
 cfg=MotionGenConfig.load_from_robot_config('franka.yml',world,TensorDeviceType(),use_cuda_graph=False,num_trajopt_seeds=4,num_graph_seeds=4,num_ik_seeds=16,interpolation_dt=.01)
 planner=MotionGen(cfg)
 q=planner.get_retract_config().view(1,-1)
 start=JointState.from_position(q)
 pose=planner.rollout_fn.compute_kinematics(start).ee_pose
 goal=Pose(position=pose.position+torch.tensor([[.04,0.,.03]],device='cuda'),quaternion=pose.quaternion)
 result=planner.plan_single(start,goal,MotionGenPlanConfig(max_attempts=2,enable_graph=True))
 torch.cuda.synchronize()
 assert bool(result.success.item()),str(result.status)
 traj=result.get_interpolated_plan();assert torch.isfinite(traj.position).all() and len(traj.position)>1
 end=planner.rollout_fn.compute_kinematics(JointState.from_position(traj.position[-1:].contiguous())).ee_pose
 error=float(torch.linalg.norm(end.position-goal.position).item());assert error<.01
 torch.save({'positions':traj.position.cpu(),'velocities':traj.velocity.cpu(),'goal_position':goal.position.cpu(),'goal_quaternion':goal.quaternion.cpu()},out/'trajectory.pt')
 report={'passed':True,'position_error_m':error,'steps':len(traj.position),'wall_seconds':time.monotonic()-t0,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
 (out/'report.json').write_text(json.dumps(report,indent=2));print('CUROBO_PLAN_PASS '+json.dumps(report),flush=True)
except BaseException:
 (out/'error.json').write_text(json.dumps({'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-t0},indent=2));raise
