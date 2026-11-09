"""Versioned benchmark contracts; explicit transforms and provenance, no silent units."""
from copy import deepcopy
from pathlib import Path,PurePosixPath
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from adapters.single_arm.contract import pose_matrix,rigid_matrix,transform_pose,RobotAdapter

IDS={'milk','tissue','desk','tray','divider','monitor'}

def local_asset(path):
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or not p.suffix:raise ValueError('Asset must be a relative contained file')
    return str(p)

def make_specs(source,metadata):
    if source.get('units')!='m' or source.get('up_axis')!='Z':raise ValueError('Source must explicitly declare m/Z')
    T=np.eye(4);T[2,3]=.75
    objects=[]
    for o in source['objects']:
        objects.append({'id':o['id'],'asset':'assets/photo/'+str(PurePosixPath(o['asset']).relative_to('assets')),'prim_path':o['prim_path'],
          'pose_world':transform_pose(o['position_m']+o['quaternion_wxyz'],T),'size_m':deepcopy(o['size_m']),
          'dynamic':metadata['objects'][o['id']]['dynamic'],'physics':deepcopy(o['physics']),'evidence':deepcopy(o['evidence']),
          'source_record':deepcopy(o),'scale_xyz':[1,1,1]})
    scene={'schema':'evoworld.scene.v1','mode':'reconstruction','generation_execution':'cached_phase5_gpt_assets_no_new_inference','units':'m','up_axis':'Z','source_frame':'photo_tabletop_anchor','world_frame':'simulation_floor','world_T_source':T.tolist(),'objects':objects,
        'provenance':{'world_T_source':{'status':'designed','value':'tabletop at0.75m, no measured floor height'},'source_geometry':'copied unchanged from acceptedPhase5 round3; per-object evidence retained','randomization':'none'},'dynamic_ids':['milk','tissue'],'builtin_runtime_resources':['OmniPBR.mdl']}
    base=[0,-.48,.001,2**-.5,0,0,2**-.5]
    robot={'schema':'evoworld.robot.v1','id':'franka','type':'SAGE_fixed_base_Panda','asset':'robot/scene.usda','prim_path':'/panda','base_pose_source':base,'base_pose_world':transform_pose(base,T),'joint_names':RobotAdapter.joint_names,'default_q':[0,-.569,0,-2.810,0,3.037,.741,.04,.04],'tool_frame':'panda_hand','tool_offset_m':[0,0,0],'action_schema':RobotAdapter.schema,'fixed_base':True,
        'provenance':{'base_pose_world':{'status':'designed','reason':'simulation placement at front of inferred desk, facing+Y; not observed in userphotos'},'default_q':{'status':'copied','source':'SAGE FRANKA_PANDA_HIGH_PD_CFG'},'robot_geometry':{'status':'copied','source':'cached official SAGE Isaac4.2 Panda'}}}
    cameras={'left':{'eye':[.65,-.65,1.25],'target':[0,.02,.88]},'right':{'eye':[-.65,-.55,1.25],'target':[0,.02,.88]},'overview':{'eye':[1.05,-1.35,1.65],'target':[0,-.10,.93],'resolution_wh':[960,720]},'detail':{'eye':[.58,-.58,1.30],'target':[0,0,.84],'resolution_wh':[640,480]}}
    task={'schema':'evoworld.task.v1','name':'photo_tabletop_franka_load_and_motion','description':'Load cached photo reconstruction in SAGE and verify Franka reset, observation and short free-space movement','seed':31,'target_object':'milk','physics_dt_s':.01,'control_dt_s':.02,'settle_steps':100,'motion_steps_each':30,'motion_offset_base_m':[0,0,.03],'gripper':1,'cameras':cameras,'acceptance':{'endpoint_error_m':.005,'object_tail_span_m':.001,'object_tail_speed_m_s':.01,'support_error_m':.003,'unintended_contact_N':1.0},'provenance':{'cameras':{'status':'designed','reason':'simulation inspection views, not fitted realcamera'},'motion_and_acceptance':{'status':'designed','reason':'bounded functional integration smoke'},'randomization':'none'},'grasp_verified':False,'real_robot':False}
    validate_specs(scene,robot,task)
    return scene,robot,task

def validate_specs(scene,robot,task):
    if (scene.get('schema'),robot.get('schema'),task.get('schema'))!=('evoworld.scene.v1','evoworld.robot.v1','evoworld.task.v1'):raise ValueError('Unknown schema')
    if scene.get('units')!='m' or scene.get('up_axis')!='Z' or scene.get('mode')!='reconstruction':raise ValueError('Explicit reconstruction/m/Z required')
    if scene.get('generation_execution')!='cached_phase5_gpt_assets_no_new_inference' or robot.get('prim_path')!='/panda' or robot.get('tool_offset_m')!=[0,0,0] or task.get('gripper')!=1:raise ValueError('Unsupported fixed-profile generation/robot/tool/gripper setting')
    T=rigid_matrix(scene['world_T_source']);objects=scene['objects'];ids=[o['id'] for o in objects]
    if set(ids)!=IDS or len(ids)!=len(IDS):raise ValueError('Benchmark requires exactly6unique accepted assets')
    if set(scene['dynamic_ids'])!={'milk','tissue'}:raise ValueError('Dynamic IDs changed')
    for o in objects:
        if o.get('prim_path')!='/Asset':raise ValueError('Fixed profile expects native /Asset default prim')
        local_asset(o['asset']);pose_matrix(o['pose_world']);original=o['source_record']
        expected=transform_pose(original['position_m']+original['quaternion_wxyz'],T)
        if not np.allclose(pose_matrix(o['pose_world']),pose_matrix(expected),atol=1e-8):raise ValueError('Source transform applied incorrectly')
        if o['scale_xyz']!=[1,1,1] or o['size_m']!=original['size_m'] or o['physics']!=original['physics'] or o['evidence']!=original['evidence']:raise ValueError('Source geometry/physics/provenance changed')
        if o['dynamic']!=(o['id'] in scene['dynamic_ids']):raise ValueError('Wrong rigid role')
    local_asset(robot['asset']);pose_matrix(robot['base_pose_world']);RobotAdapter().check_joints(robot['joint_names'])
    if not np.allclose(pose_matrix(robot['base_pose_world']),pose_matrix(transform_pose(robot['base_pose_source'],T)),atol=1e-8):raise ValueError('Robot base transform mismatch')
    if robot.get('provenance',{}).get('base_pose_world',{}).get('status')!='designed':raise ValueError('Robot placement must be attributed as designed')
    if scene['provenance']['world_T_source']['status']!='designed':raise ValueError('World placement provenance missing')
    if not robot['fixed_base'] or robot['tool_frame']!='panda_hand' or robot['action_schema']!=RobotAdapter.schema:raise ValueError('Unsupported robot interface')
    q=np.asarray(robot['default_q']);offset=np.asarray(task['motion_offset_base_m'])
    if q.shape!=(9,) or not np.isfinite(q).all() or offset.shape!=(3,) or not np.isfinite(offset).all() or np.linalg.norm(offset)>.05:raise ValueError('Invalid reset or bounded motion')
    if task['physics_dt_s']!=.01 or task['control_dt_s']!=.02 or task['settle_steps']!=100 or task['motion_steps_each']!=30:raise ValueError('Unsupported benchmark timing')
    if task['real_robot'] or task['grasp_verified']:raise ValueError('This contract does not certify hardware/grasp')
    return True

def action_record(action):
    """Convert validated float32/64 targets to portable JSON scalar values."""
    return RobotAdapter().action(action).tolist()
