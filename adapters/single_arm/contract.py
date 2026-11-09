"""Strict Phase6 simulation contracts. No G1 controls, API, planners or implicit units."""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation


def pose_matrix(pose):
    p=np.asarray(pose,dtype=float)
    if p.shape!=(7,) or not np.isfinite(p).all() or abs(np.linalg.norm(p[3:])-1)>1e-5:
        raise ValueError('Expected finite XYZ,wxyz with unit quaternion')
    T=np.eye(4);T[:3,:3]=Rotation.from_quat(p[[4,5,6,3]]).as_matrix();T[:3,3]=p[:3]
    return T


def rigid_matrix(matrix):
    T=np.asarray(matrix,dtype=float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1],atol=1e-8) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-8):
        raise ValueError('Frame transform must be SE(3), no scaling/reflection')
    return T


def transform_pose(pose, target_T_source):
    T=rigid_matrix(target_T_source)@pose_matrix(pose)
    q=Rotation.from_matrix(T[:3,:3]).as_quat()
    return np.r_[T[:3,3],q[3],q[:3]].tolist()


class RobotAdapter:
    schema='evoworld.sage_franka.v1'
    joint_names=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
    def __init__(self,env=None):
        self.env=None;self.previous=None
        if env is not None:self.bind(env)
    def bind(self,env):
        self.check_joints(env.scene['robot'].joint_names)
        robot=env.scene['robot']
        if not robot.is_fixed_base or robot.body_names[8]!='panda_hand':raise ValueError('Original Franka body/frame contract mismatch')
        self.env=env;self.previous=None
        return self
    def reset(self,seed=31):
        if self.env is None:raise RuntimeError('Bind a live SAGE Franka simulation first')
        self.env.reset(seed=seed)
        robot=self.env.scene['robot']
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        self.env.scene.write_data_to_sim();self.previous=None
    def step(self,action):
        if self.env is None:raise RuntimeError('Bind a live SAGE Franka simulation first')
        import torch
        a=self.action(action,previous=self.previous)
        result=self.env.step(torch.tensor(a[None],device=self.env.device,dtype=torch.float32))
        self.previous=a.copy()
        return result
    def check_schema(self,schema):
        if schema!=self.schema:raise ValueError('Unsupported robot contract; explicit migration required')
    def check_joints(self,names):
        if list(names)!=self.joint_names:raise ValueError('Joint order/name mismatch')
    def action(self,action,*,units='m',frame='robot_base',previous=None):
        a=np.asarray(action,dtype=float)
        if units!='m' or frame!='robot_base':raise ValueError('Expected metres in robot_base')
        if a.shape!=(8,) or not np.isfinite(a).all():raise ValueError('Expected finite8D hand pose + sign')
        pose_matrix(a[:7])
        if np.any(a[:3]<[-1,-1,-.2]) or np.any(a[:3]>[1,1,1.5]) or a[7] not in (-1,1):raise ValueError('Action workspace/sign violation')
        if previous is not None:
            p=self.action(previous)
            angular=(Rotation.from_matrix(pose_matrix(p[:7])[:3,:3]).inv()*Rotation.from_matrix(pose_matrix(a[:7])[:3,:3])).magnitude()
            if np.linalg.norm(a[:3]-p[:3])>.0300001 or angular>.250001:raise ValueError('Per-step target jump exceeds simulation guardrail')
        return a.copy()
    def snapshot(self,state):return deepcopy(state)
    def restore(self,snapshot):return deepcopy(snapshot)
    def validate_observation(self,obs):
        expected={'joint_pos':((9,),'float32'),'joint_vel':((9,),'float32'),'joint_qpos':((8,),'float32'),'ee_pose':((8,),'float32'),'actions':((8,),'float32'),'points':((10000,3),'float32')}
        for view in ['left','right','wrist']:
            expected['rgb_'+view]=((128,128,3),'uint8');expected['depth_'+view]=((128,128,1),'uint8')
        if set(obs)!=set(expected):raise ValueError('Observation fields differ from documented contract')
        for k,(shape,dtype) in expected.items():
            a=np.asarray(obs[k])
            if a.shape!=shape or str(a.dtype)!=dtype or not np.isfinite(a).all():raise ValueError('Observation shape/dtype/nonfinite: '+k)
        pose_matrix(obs['ee_pose'][:7])
        return {k:{'shape':list(np.asarray(v).shape),'dtype':str(np.asarray(v).dtype)} for k,v in obs.items()}


class TaskAdapter:
    labels={'load':'load_error','reset':'reset_mismatch','motion':'tracking_error','contact':'contact_not_established','release':'release_failed','observation':'observation_contract','timeout':'timeout','planning':'planning_failed','grasp':'no_lift','place':'placement_error'}
    def __init__(self,robot=None):self.robot=robot
    def reset(self,seed=31):
        if self.robot is None:raise RuntimeError('Task requires a bound RobotAdapter')
        return self.robot.reset(seed=seed)
    def failure_label(self,stage):
        if stage not in self.labels:raise ValueError('Unknown failure stage')
        return self.labels[stage]
    def contact_success(self,blocked_finger,empty_finger,force_newton,reopened_finger):
        a=np.asarray([blocked_finger,empty_finger,force_newton,reopened_finger],float)
        return bool(np.isfinite(a).all() and blocked_finger>.01 and blocked_finger-empty_finger>.01 and force_newton>.1 and reopened_finger>.035)


def _scene(objects,T,source):
    T=rigid_matrix(T);ids=[x['id'] for x in objects]
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('Need unique nonempty object IDs')
    return {'schema':'evoworld.scene.v0.1','units':'m','up_axis':'Z','source':source,'world_T_source':T.tolist(),'objects':objects,'robot_control_imported':False}


def scene_from_realtosim(spec,world_T_source):
    if spec.get('units')!='m' or spec.get('up_axis')!='Z':raise ValueError('Explicit m/Z required')
    objects=[]
    for o in spec['objects']:
        size=np.asarray(o['size_m'],float)
        if size.shape!=(3,) or not np.isfinite(size).all() or np.any(size<=0):raise ValueError('Invalid size')
        objects.append({'id':o['id'],'asset':o['asset'],'pose_world':transform_pose(o['position_m']+o['quaternion_wxyz'],world_T_source),'size_m':size.tolist(),'physics':deepcopy(o['physics']),'evidence':deepcopy(o['evidence']),'source_record':deepcopy(o)})
    return _scene(objects,world_T_source,'g1-realtosim generic assets only')


def scene_from_sage(transforms,properties,world_T_source):
    objects=[]
    for name,o in transforms.items():
        p=[o['position'][k] for k in 'xyz'];q=Rotation.from_euler('xyz',[o['rotation'][k] for k in 'xyz'],degrees=True).as_quat()
        prop=deepcopy(properties[name])
        objects.append({'id':name,'asset':name+'.usdz','pose_world':transform_pose(p+[q[3],*q[:3]],world_T_source),'physics':prop,'evidence':{'pose':'SAGE export: position metres, extrinsic xyz Euler degrees','physics':'source export; not newly calibrated'},'source_record':deepcopy(o)})
    return _scene(objects,world_T_source,'SAGE exported object transforms; native USD local geometry preserved')
