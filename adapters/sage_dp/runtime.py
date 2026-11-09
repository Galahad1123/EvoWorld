"""Runtime instrumentation only; no grasp/planner or collector algorithm changes."""
import json
from pathlib import Path
import numpy as np
import torch


def cpu(x):
    return x.detach().cpu().numpy()


def configure(cfg, seed):
    cfg.seed = seed
    cfg.scene.ee_frame.debug_vis = False
    cfg.scene.camera_wrist.prim_path = '{ENV_REGEX_NS}/camera_wrist'
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.concatenate_terms = False
    # Auxiliary videos are not policy observations. Keep policy RGB at 128x128.
    for key, sensor in vars(cfg.scene).items():
        if key.startswith('camera_') and '_vis' in key and sensor is not None:
            sensor.width = 320
            sensor.height = 192
    return cfg


def attach(env):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'single_arm'))
    from contract import transform_pose, pose_matrix
    robot = env.scene['robot']
    hand = robot.body_names.index('panda_hand')
    assert hand == 8
    offset = env.cfg.scene.camera_wrist.offset
    original_render = env.sim.render

    def render(*args, **kwargs):
        link = cpu(robot.root_physx_view.get_link_transforms()[0, hand])
        pose = np.r_[link[:3], link[6], link[3:6]]
        optical = transform_pose(list(offset.pos) + list(offset.rot), pose_matrix(pose))
        env.scene['camera_wrist'].set_world_poses(
            torch.as_tensor(optical[:3], device=env.device, dtype=torch.float32)[None],
            torch.as_tensor(optical[3:], device=env.device, dtype=torch.float32)[None], convention='ros')
        return original_render(*args, **kwargs)
    env.sim.render = render


def snapshot(env, attempt, target, place, out):
    robot = env.scene['robot']
    record = {'attempt': attempt, 'robot_root': cpu(robot.data.root_state_w).tolist(),
              'q': cpu(robot.data.joint_pos).tolist(), 'qd': cpu(robot.data.joint_vel).tolist(),
              'previous_action': cpu(env.action_manager.action).tolist(),
              'objects': {k: cpu(v.data.root_state_w).tolist() for k, v in env.scene.rigid_objects.items()},
              'target': target, 'place_world': cpu(place).tolist(), 'cameras': {}}
    for name in ['left', 'right', 'wrist']:
        cam = env.scene['camera_' + name]
        record['cameras'][name] = {'position': cpu(cam.data.pos_w).tolist(),
                                  'quaternion': cpu(cam.data.quat_w_ros).tolist(),
                                  'intrinsics': cpu(cam.data.intrinsic_matrices).tolist()}
    Path(out, f'initial-{attempt:03}.json').write_text(json.dumps(record, indent=2))


def trace(env, attempt, iteration, target, actions, out):
    obj = env.scene[target]
    r = {'attempt': attempt, 'iteration': iteration, 'object': cpu(obj.data.root_com_state_w[0]).tolist(),
         'action': cpu(actions[0]).tolist(), 'q': cpu(env.scene['robot'].data.joint_pos[0]).tolist()}
    with Path(out, 'physical-trace.jsonl').open('a') as f:
        f.write(json.dumps(r) + '\n')
