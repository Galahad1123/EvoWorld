# Phase 6: single-arm interface and compatibility audit

Accepted on5090-1: `experiments/phase6-single-arm-WdbNEP0U`. Original SAGE fixed-base Franka Panda loads in IsaacSim6.0.1, performs two repeatable free-space movements and a measured finger-contact/release test through `RobotAdapter` and `TaskAdapter`. The original M2T2, cuRobo and SAGE task source are unchanged. This phase uses a small diagnostic fixture; integrating the Phase5 photographed scene with the arm remains Phase7.

## Implemented interface

`contract.py` provides `RobotAdapter.bind(env)`, `reset(seed)`, `step(action)`, strict observation/action validation, named joint checks, explicitSE(3) pose conversion and source-scene mappings. `TaskAdapter.reset()` delegates the deterministic environment reset and provides the contact/release success predicate and failure labels. No G1 controller is imported. `snapshot/restore` are defensive CPU value copies, not a physics-state restore API; live reset resets configured defaults through SAGE. The implementation currently targets one simulation environment and one fixed-base Panda.

| Item | Contract |
|---|---|
| Robot | Original cached instanceable Panda USD under `assets/isaac42/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd` |
| Planner description | Existing cuRobo `franka.yml`, `franka_description/franka_panda.urdf`; base `panda_link0`, goal `panda_hand`; fingers locked0.04m for planning |
| Joint order | panda_joint1..7, panda_finger_joint1, panda_finger_joint2; 7revolute radians +2prismatic metres |
| Default configuration | [0,−0.569,0,−2.810,0,3.037,0.741,0.04,0.04] |
| Base | Fixed, audit origin XYZ0,wxyzidentity; arbitrary mapping requires explicit `world_T_source`, not implicit translation |
| Tool | `panda_hand`, zero control offset; live body index8 asserted against its name. Contact fixture center +0.1034m along handZ is a designed test point, not a redefinition of commanded tool |
| Action | float8: XYZ metres, quaternionwxyz, gripper +1open/−1close, absolute hand pose in robot-base frame |
| Control | Original absolute differentialIK/DLS + implicit PD; physics100Hz, control/observations50Hz simulated |
| Adapter guards | finite values, normalized quaternion, XYZbox[−1,−1,−0.2]..[1,1,1.5]m; target translation≤30mm and rotation≤0.25rad between consecutive commands; no silent clipping |
| Reset | SAGE seeded reset restores configured root/joint/rigid defaults, velocities, action/episode/sensor buffers; joint targets reset to defaults; adapter previous-action history cleared; audit resets external camera views |

Live joint limits (rad for1–7, m forfingers) match cuRobo URDF: ±2.8973;±1.7628;±2.8973;[−3.0718,−0.0698];±2.8973;[−0.0175,3.7525];±2.8973;[0,0.04]each finger. SAGE shoulder/forearm effort limits87/12Nm, velocity2.175/2.61rad/s, highPD400/80. Finger simulation limit200N, stiffness2000/damping100, velocity0.2m/s. **URDF finger effort20N differs from SAGE simulation200N**; preserve existing simulation baseline, do not interpret it as hardware capability. Guardrails are simulation checks, not IK feasibility, collision-free guarantees or certified hardware safety. Unmodified original SAGE collection does not pass through this additional adapter.

## Observation and timing contract

All12 policy fields were validated during every one of270 steps. Shapes below omit batch1:

| Field | Shape/type | Meaning |
|---|---|---|
| joint_pos |9,float32|Joint positions relative to configured defaults; armrad/fingerm |
| joint_vel |9,float32|Relative to configured default velocities; rad/s,m/s |
| joint_qpos |8,float32|Seven absolute arm positions + sign(gripper action+0.02); not8jointangles |
| ee_pose |8,float32|Named hand pose expressed in base + sign(last finger position−0.03); measured sign can be0 at boundary |
| actions |8,float32|Action manager's most recently applied target |
| rgb_left/right/wrist |128×128×3,uint8|Rendered RGB |
| depth_left/right/wrist |128×128×1,uint8|Original depth clipped to2m then scaled0–255; invalid→0. Quantized, not raw metric depth |
| points |10000×3,float32|Base-frame XYZ, randomly resampled each observation; not deterministic point order |

Cameras: original focal10mm/aperture20.955mm; K=[[61.083275,0,64],[0,61.083275,64],[0,0,1]],128², clipping[0.001,1000]m. Optical frame is ROS/CV(+Zforward,−Yup); USD/OpenGL conversion is diag(1,−1,−1). Audit left/right eyes[0.8,±0.65,0.65], look-at[0.4,0,0.3]; exact world poses/quaternions are in `runtime.json`. Wrist retains original local mount[0,0,0.08], quaternion[0,0,0,1]wxyz relative tohand (verified runtime values are authoritative).

`trace.json` records global simulation time, episode control-step index, sensor clock since reset, frame counters, command, joint state, hand pose and contact force. Global simulation time continues through reset; camera clocks reset. Recorded step spacing is20ms; camera clock corresponds to step×0.02. These are simulation/software sampling timestamps, not hardware exposure timestamps. Original Phase3 HDF did not save these clocks/calibration; they are not retroactively added. No action-delay injection; commands apply on the next physics substeps. Wall time median/p95 perstep is measured in `readback.json`; simulated50Hz is not a real-time guarantee.

## Compatibility correction

Initial audit `dVXtvFlJ` passed robot motion/contact but failed wrist attachment: camera position differed from live hand attachment by0.56869m. Camera code uses legacy `XFormPrim.get_world_poses()` (defaultUSD); articulated PhysX state and nested camera USD pose disagreed. `krbPS7zM` tested disablingFabric; it did not solve it. Both are failed camera acceptance runs.

The adapter runtime keeps the original optical mount but creates the wrist camera at a world-level path and synchronizes its pose from live PhysX link transforms immediately before each render. Fabric remains enabled. Debug frame markers are disabled in this fixture. No upstream or Phase3 source is patched. `JKQxLuF9` proved the correction; `WdbNEP0U` revalidated it through the final reusable reset/step interface and launcher.

Free-space wrist imagery is informative after correction. The final contact-view wrist image is black: the original0.08m mount is extremely close to the test block (nominal front face at handZ0.0809m, near clip1mm), so contact-time visibility is not accepted for policy learning. This does not affect measured force/closure acceptance. Camera placement/occlusion remains a task-specific requirement for Phase7/8; no training-quality imagery claim.

## Scene bridge and task boundary

`map_scenes.py` mapped actual Phase5 six-asset spec and actual SAGE four-object export into provisional `evoworld.scene.v0.1`, preserving source records, physics/evidence and explicit frame transforms. SAGE exported XYZposition and extrinsicXYZ Euler degrees become quaternionwxyz; g1 generic USD paths/poses remain generic assets. Identity mappings preserve two **distinct** frames (`photo_tabletop_anchor`, `sage_room_floor`); they do not align the scenes. Asset paths resolve against source-file parents; this is a representation audit, not a new portable bundle or robot/photo scene assembly. Full canonical Task/Scene/RobotSpec integration remainsPhase7. G1 locomotion/state/action/HDF and Phase5 object-only HDF are rejected as Franka demonstrations.

Audit success: both30mm vertical motions finish within5mm; repeated robot trajectories agree≤1e-6; reset states match; close against a50mm wide block yields single-finger opening>10mm and >10mm more than empty closure, contact force>0.1N, reopen>35mm. The block is kinematic; this demonstrates contact, not lifting. Failure labels: load_error,reset_mismatch,tracking_error,contact_not_established,release_failed,observation_contract,timeout,planning_failed,no_lift,placement_error. Original SAGE pick/place success remains its separate state machine (EEposition<3cm, actual lift>60%objectheight, placementXY<8cm thenrelease); no new pick/place episode or original planner call occurred inPhase6. Historical Phase3 M2T2→cuRobo cube success remains supporting baseline evidence.

## Verification and reproduction

Final run:270 control states,540physics steps,5.4s simulated. Repeat joint difference0; endpoints≈2.84/2.79mm. Empty finger≈0.112mm, blocked≈25.126mm, both-finger contact force minimum last5steps≈6.98N, reopen≈39.998mm. Wrist attachment error≈3.9e−9m.19 CPU tests pass, including mutated actual trace negatives for stale wrist, bad camera clock, missing contact, reset mismatch, reordered joints and incomplete trace. Four bounded GPU processes total, no API/newweights/training. Early failures are retained.

On5090-1 from `~/lizk/vlas/EvoWorld`:

```bash
# Creates a fresh experiment, captures input/source hashes, caps simulator180s,
# then fails closed on simulator status, independent readback or tests.
bash adapters/single_arm/run.sh

# CPU-only recheck of accepted evidence:
envs/g1-realtosim/bin/python adapters/single_arm/readback.py experiments/phase6-single-arm-WdbNEP0U
EVOWORLD_PHASE6_ACCEPTED="$PWD/experiments/phase6-single-arm-WdbNEP0U" \
  envs/g1-realtosim/bin/python -m unittest discover -s adapters/single_arm -v
```

The launcher sources only existing mirror/CUDA paths, not `.env`. Accepted run includes pre-run source hashes, runtime robot/camera contract, traces, images, observations, actual scene mappings and CPU logs. Runtime dependencies are the already installed `sage-robot` overlay for simulation and `g1-realtosim` for CPU checks; no environment changes were needed.

## Final review and source history

Fresh independent review found no Critical/Important issue. Three minor findings were closed with explicit scene-unit requirements, all-step wrist position/final-orientation gates, and corrected ledger semantics.19 final CPU tests pass; regression tests first demonstrated the three missing gates. No additional GPU run was needed. `execution-sources/` preserves all24 pre-run hashed files from the actual simulator execution; validation-only hardening afterward changed `contract.py`, `readback.py` and tests, without changing accepted action/reset/motion/camera execution behavior. `validation-sources-sha256.json` identifies the final validators. The launcher still requires a fresh execution plus CPU acceptance for a future run.
