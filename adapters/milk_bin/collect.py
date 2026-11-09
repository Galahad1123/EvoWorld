"""Run the original SAGE collector with explicit native-scene adapter hooks."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
from evoworld.config import resource
source=resource('expert_source')/'server/isaaclab/data_generation_franka_pnp_from_layout_parsing.py'
text=source.read_text();seed=int(os.environ.get('EVOWORLD_DP_SEED','308'))
changes=[
 ('                T_reach_place_envs = {}','                T_reach_place_envs = {}\n                release_hold_pose_envs = {}'),
 ('                            T_reach_place_envs[env_i] = iteration','                            T_reach_place_envs[env_i] = iteration\n                            release_hold_pose_envs[env_i] = current_ee_pose[env_i, :7].clone()'),

 ('                        if is_reach_place_goal:\n                            gripper_switch = 1', '                        if is_reach_place_goal:\n                            ee_goal = release_hold_pose_envs[env_i]\n                            gripper_switch = 1 if iteration - T_reach_place_envs[env_i] >= photo.dwell_steps() else -1'),
 ('                            place_traj_envs[env_i] = place_traj_envs[env_i].reshape(-1, 1, 7).repeat(1, 2, 1).reshape(-1, 7)', '                            n_pause = photo.dwell_steps()\n                            place_traj_envs[env_i] = torch.cat([place_traj_envs[env_i][:n_pause], place_traj_envs[env_i][n_pause:].reshape(-1, 1, 7).repeat(1, 2, 1).reshape(-1, 7)], dim=0)'),

 ('                            target_to_ee_pos[2] = 0', '                            ee_goals_place[env_i, 2] = place_location_r[env_i, 2] - target_object_height - 0.15 + target_to_ee_pos[2] + 0.075'),
 ('is_target_reach = torch.norm(target_pos_xy_w - place_xy_w) < 0.08','is_target_reach = torch.norm(target_pos_xy_w - place_xy_w) < 0.012'),

 ('import random\n',f'import random\nrandom.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed})\nimport milk_runtime as photo\n'),
 ('    all_object_rigids = list(mass_dict.keys())','    all_object_rigids = ["milk", "tissue"]'),
 ('    camera_pos, camera_lookat = get_default_camera_view()','    object_transform_dict = photo.transforms()\n    camera_pos, camera_lookat = get_default_camera_view()'),
 ('    env = gym.make(args_cli.task, cfg=env_cfg).env',f'    photo.configure(env_cfg,{seed})\n    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)\n    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)\n    env = gym.make(args_cli.task, cfg=env_cfg).env\n    photo.attach(env)'),
 ('                ee_goals_lift_up[:, 2] = place_location_r[:, 2]','                ee_goals_lift_up[:, 2] = torch.maximum(place_location_r[:, 2], ee_goals[:, 2] + expected_success_height + 0.04)'),
 ('    T_grasp = 400','    T_grasp = 650'),('    T_close_gap = 15','    T_close_gap = 19'),('    T_open_gap = 50','    T_open_gap = 150'),
 ('                # set objects','                photo.randomize(object_transform_dict)\n                # set objects'),
 ('                        place_pos_xy_w = env.scene[place_object_name].data.root_state_w[env_i, :2].reshape(2)','                        place_pos_xy_w = place_xy_w.clone()  # fixed static desk goal'),
 ('            obs_dict = env.observation_manager.compute()','            photo.capture_before_step(env,current_data_idx,iteration)\n            if iteration == T_init + 1:\n                photo.snapshot(env,current_data_idx,target_object_name,place_location_w,".")\n            obs_dict = env.observation_manager.compute()'),
 ('            dones = terminated | truncated','            dones = terminated | truncated\n            photo.trace(env,current_data_idx,iteration,target_object_name,actions,".")'),
 ('                        collector_interface.flush()','                        collector_interface.flush()\n                        with open("demo-attempts.jsonl","a") as stream:stream.write(json.dumps({"demo":collected_data_episodes,"attempt":current_data_idx-1})+"\\n")'),
 ('                # Sample grasp poses for all environments','                if os.environ.get("EVOWORLD_PREVIEW_ONLY")=="1":\n                    photo.snapshot(env,current_data_idx,target_object_name,place_location_w,".")\n                    env.close();simulation_app.close();raise SystemExit(0)\n                # Sample grasp poses for all environments'),
 ('                    if is_reach_place_goal:\n                        for data_i','                    if is_reach_place_goal and photo.accept_demo(current_data_idx-1, min(T_reach_place_envs[env_i] + T_open_gap, T_grasp)):\n                        for data_i'),
 ('    # run the main function','    photo.install(globals())\n    # run the main function')]
for old,new in changes:
 assert text.count(old)==1,(old,text.count(old))
 text=text.replace(old,new)
Path('collector-executed.py').write_text(text)
Path('collector-source.json').write_text(json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'executed_sha256':hashlib.sha256(text.encode()).hexdigest(),'seed':seed,'changes':[a for a,b in changes]},indent=2))
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
