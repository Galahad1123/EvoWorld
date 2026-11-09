# Franka task compatibility baseline

Run on 5090-1 from the EvoWorld root:

```bash
# Bounded official asset cache, through existing local proxy; no API credential needed.
envs/isaacsim6/bin/python adapters/franka_baseline/fetch_assets.py
# One attempt, 300 s timeout; creates experiments/phase3-franka-*/.
envs/sage/bin/python adapters/franka_baseline/launch.py
# Use the emitted run path:
envs/isaacsim6/bin/python adapters/franka_baseline/read_trajectory.py RUN/episode/trajectory.hdf5
envs/sage/bin/python -m pytest adapters/franka_baseline/tests -q
```

This baseline combines a real GPT-generated SAGE room (30dd907ee170), the Isaac Sim 6 Franka Panda asset, a new standard dynamic cube and Isaac's RMPflow pick-place controller. No scene edits or policy choices are made by a second LLM; no new GPT call is needed for this deterministic control test. This is not the original SAGE IsaacLab/M2T2/cuRobo generator and does not establish trained-policy performance.

The original SAGE Franka entry point is `server/isaaclab/data_generation_franka_pnp_from_layout_parsing.py` at commit fe3966013a495dc1642710de93a83ddf66e534a0. It imports old `omni.isaac.lab`, M2T2, cuRobo, Open3D and igl. Its task registration is `Isaac-Lift-Single-Obj-Scene-Franka-IK-Abs-v0-vis-teaser`; the bundled IsaacLab uses `FRANKA_PANDA_HIGH_PD_CFG` and `panda_instanceable.usd`. Our installed SDK uses a different Panda USD and gains/controller. This difference is intentional and must not be described as an exact reproduction.

## Data contract

`data/demo_0` in `trajectory.hdf5` holds N transitions. The data group has `total` and JSON `env_args`; the episode has `num_samples` and `success` attributes.

| Field | Shape | Meaning |
| --- | --- | --- |
| actions | N × 18 | q[0:9]: 7 arm joints in radians + 2 finger joints in meters; qdot[9:18]: rad/s + m/s. Absent positions hold their previous command; absent velocity targets become zero. Finger position targets clipped to [0,0.04] m. |
| obs/joint_pos, next_obs/joint_pos | N × 9 | Actual measured joint positions before/after the same applied action. |
| obs/joint_vel, next_obs/joint_vel | N × 9 | Actual velocities, rad/s and m/s. |
| obs/object_pos, next_obs/object_pos | N × 3 | Cube center in world frame, meters. |
| obs/object_quat, next_obs/object_quat | N × 4 | Cube orientation, wxyz. |
| obs/object_vel, next_obs/object_vel | N × 3 | Cube linear velocity, m/s. |
| obs/ee_pos, next_obs/ee_pos | N × 3 | `panda_rightfinger` origin in world frame, meters; **not** the tool center. |
| obs/ee_quat, next_obs/ee_quat | N × 4 | Same link's world orientation, wxyz. |
| obs/target_pos, next_obs/target_pos | N × 3 | Fixed desired cube center. |
| rewards | N | Negative Euclidean cube-to-target distance after the action; diagnostic reward, not original SAGE reward. |
| dones / terminated / truncated | N | Final row ends the recorded episode; terminated on success, truncated on failed capped/finished attempt. |
| time | N | Pre-action simulation time since recording starts, seconds; 60 Hz. |
| controller_event | N | Built-in controller phase before each action. |

Meters, Z-up, right-handed world. Joint order is recorded in `task.json`. Robot base is fixed at [1.45,1.97,0.8] m, interpreted as a table-height mounted arm; the mounting hardware is not modeled. Cube side 0.04 m, mass 0.05 kg, initial center [1.95,1.80,0.825] m; target [1.95,2.20,0.825] m. The table/chairs/mug are retained from the SAGE scene, with physical contacts enabled. No object teleportation or attachment is used during the recorded rollout.

Reset calls `world.reset()` and controller reset. A deliberate pre-recording cube perturbation is followed by reset and numerical checks, then 60 unrecorded settling steps. Explicit slower phase durations [180,240,30,60,120,180,240,60,120,60] steps; cube contact solver 32 position/8 velocity iterations. Maximum 1800 recorded steps (30 simulated seconds). The recorder independently checks physics callback count, step count and elapsed simulation time. Rendering is asserted not to advance that clock.

Success requires: controller completion, measured lift >= 0.05 m above the first recorded cube center, target error <= 0.03 m over the last 61 states (1 s), both finger joints > 0.025 m throughout that tail, object speed < 0.03 m/s and per-step translation < 0.002 m. Failed attempts remain in experiments and are not silently excluded. No statistical success rate follows from one episode.

The strict reader validates all dimensions, finiteness, quaternions, transition continuity, rewards, end flags and recomputed physical success. The upstream `read_pick_n_place_data.py` is only a shallow h5py hierarchy inspector: passing it proves structural readability, not compatibility with SAGE training or Robomimic policies. No RL learner is installed/validated in this phase.

## Original SAGE contracts and remaining integration work

The original generator creates 8-D actions: absolute end-effector xyz + wxyz in robot-base coordinates plus binary gripper command. Its differential IK controls `panda_hand`; offset/EE conventions must be matched before sharing actions. The base environment exposes relative joint position/velocity, `ee_pose`, absolute `joint_qpos`, previous actions, and visual terms in the teaser config. It resets to scene defaults. Reward terms in `lift_single_obj_scene_env_cfg.py` are commented out; default timeout is removed by the generator (`env_cfg.terminations.time_out=None`). The generator drives a 400-iteration state machine (T_init=20), uses M2T2 grasps and cuRobo paths, and filters collected demonstrations by its own reach/lift/place criteria. It records observations before the action, next observations after it, rewards and dones via `RobomimicDataCollector`.

Generic reusable pieces: scene import/assets, explicit physics stepping, provenance, bounded launch, dataset hierarchy and transition validation. Robot-specific pieces: robot asset and gains, articulation names/order, gripper limits/sign, controller/IK, base/tool transforms, reachability and task success thresholds. Switching to a target real single arm requires these adapters and a fresh acceptance test.

`task-scene.usda` is an on-host diagnostic with absolute robot/source references. It is not a standalone portable package. Initial/final PNGs are actual rendered task states; no reconstructed playback is used as physical evidence.
