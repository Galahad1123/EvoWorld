# Original SAGE robot/task baseline on 5090-1

The bounded fixed-base Franka baseline passed using **M2T2 → cuRobo → SAGE IsaacLab state machine → original RobomimicDataCollector**. GPT is not called by this robot baseline. The earlier RMPflow experiment is historical and is not its acceptance evidence.

Run `experiments/sage-original-task-LlLIdyMg` collected one successful episode on its first attempt: 202 transitions, 8-D actions, 12 observation fields. The control fixture is a procedural 5 cm / 80 g cube in the existing Phase 2 room. Its provenance explicitly distinguishes it from an AI-generated asset. The original mug fixture failed to lift in two attempts plus one diagnostic attempt; these failures remain recorded. This is functional baseline acceptance, not general grasp reliability or full SAGE deployment acceptance.

## Run and read back

On 5090-1, from `~/lizk/vlas/EvoWorld`, using the already exported fixture:

```bash
EVOWORLD_LAYOUT_ID=layout_evoworld_cube_control \
EVOWORLD_TARGET_OBJECT=room_evoworld_30dd907ee170_cube \
EVOWORLD_MAX_TRIALS=2 bash adapters/sage_robot/run_original.sh

envs/sage/bin/python adapters/sage_robot/review_run.py \
  experiments/sage-original-task-LlLIdyMg
```

The launcher creates a fresh output directory, caps simulation at 300 s / two attempts / one collected demo, then runs CPU verification. Failures remain failures. No `.env` or API credential is loaded. For a new fixture use `export_fixture.sh`; it checks a fresh success marker and report because Kit fast shutdown can mask Python exceptions. Existing USDZ directories are refused before starting Isaac. Do not overwrite accepted evidence to rerun exports.

Historical successful run used the earlier launcher without automatic post-run verification; verification was run separately. The fail-closed wrappers were added afterward, checked with shell syntax and CPU negative cases, without a further GPU run. Historical simulator exit code was not saved and is not reconstructed.

## Original data contract

| Field | Contract |
|---|---|
| `actions` | `[x,y,z,qw,qx,qy,qz,gripper]`, absolute `panda_hand` pose in robot base frame, metres, scalar-first quaternion; +1 open / -1 close |
| `obs/joint_pos`, `joint_vel` | 9 joints relative to configured defaults, radians for arm and metres for fingers; velocities per second |
| `obs/joint_qpos` | Seven absolute arm joint positions plus gripper command sign; not an 8-joint configuration |
| `obs/ee_pose` | Original body-index-8 pose in base frame plus measured finger-open sign; do not assume this index on another robot |
| `obs/actions` | Previous applied action |
| RGB | Left/right/wrist, 128×128×3 uint8 |
| Depth | Left/right/wrist, 128×128×1 uint8; original clipping/quantization at 2 m, not raw metric depth |
| Points | 10000×3 float32, robot base frame, independently randomly resampled on each observation computation |
| `next_obs` | Same schema; all deterministic fields exactly match the next stored observation in this run; point ordering is stochastic |
| `rewards`, `dones` | Original task config has no active reward/termination terms; rewards zero and dones false. Successful collection uses the original state machine separately |

Physics dt 0.01 s, decimation 2. Original robot sampler/reset, differential IK, M2T2 candidate sorting, cuRobo approach/place planning and interpolation for descent/lift remain unchanged. Original success uses end-effector position tolerance 3 cm, actual object lift above 60% of object height, and target/place XY proximity below 8 cm, followed by release. The existing function computes orientation error but returns position success only. No new rotation or full-rate settling gate is claimed. Original collector exports the successful interval through the release gap, not the whole 400-control-step attempt.

Diagnostics every five control steps bind to HDF5 actions and absolute arm joints. Independent readback rejects missing completion, no successful episode, nonfinite actions, adjacent-observation mismatch, mismatched diagnostic actions and incomplete diagnostic coverage. The unmodified original reader executes with only its hardcoded file path replaced in memory. This does not establish policy training or replay acceptance.

## Environment and compatibility

- SAGE `fe3966013a495dc1642710de93a83ddf66e534a0`; its modified IsaacLab 1.4.1, M2T2 and server copied under `src/`. Upstream remains clean.
- `envs/sage-robot`: Python 3.12.14, isolated packages with a `.pth` exposing working `envs/isaacsim6` (Isaac 6.0.1, Torch 2.11.0+cu130). Base environment and system driver unchanged. Installed package locations/versions: `experiments/sage-original-environment/packages.json`.
- CUDA 13 compiler and headers installed into the overlay through Tsinghua wheels. `env.sh` supplies CUDA_HOME, header paths, `TORCH_CUDA_ARCH_LIST=12.0`, MAX_JOBS=4. Versioned libcudart requires local `lib/libcudart.so -> libcudart.so.13` link.
- cuRobo **v0.7.8**, SHA `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`, retains SAGE's v1 API. C++/CUDA kernels unchanged; `wp.torch.device_from_torch` replaced with Warp 1.13's `wp.device_from_torch` in both copied and installed Python code.
- PointNet++ built from bundled M2T2 source; nvdiffrast 0.4.0 SHA `253ac4fcea7de5f396371124af597e6cc957bfae`. Native installs used `uv pip install --python envs/sage-robot/bin/python --no-deps --no-build-isolation PATH`; no replacement torch installation.
- Official M2T2 checkpoint revision `a20325dcd19cd1b838274974d0ebe35bfe383796`, downloaded from hf-mirror, SHA256 `e35c3cb11e06f46c5d406bdfc756bc06f48b256dd6d638408d9a5ff13deb97fb`. Trusted checkpoint loading explicitly restores `weights_only=False`; no Qwen/VLM inference needed.
- Original Isaac 4.2 instanceable Panda asset and dependencies cached under `assets/isaac42`: 19 files, 12,180,910 bytes. `sage-franka-manifest.json` records URLs/hashes. It is the original SAGE robot asset, not the earlier RMPflow example asset.

Authoritative complete source patches: `patches/sage-robot/sage-compat.patch`, `curobo-compat.patch`, `manifest.json`. On fresh copies of the pinned sources, apply patches from `adapters/sage_robot/src` using `patch -p1 < ...`; both pass forward dry-runs against pristine files. `patch_isaac6.py` is only the initial namespace migration, not a complete rebuild procedure. Full rebuild from an empty host was not tested.

Compatibility edits include SDK namespaces, explicit physics tensor stage IDs, camera-enable setting, removed schema guards, local asset root, scoped Franka-only task imports and lazy VLM import. Removed improved-patch-friction setting is accepted only for the old default enabled value; unsupported false raises. Unused deformable annotations are deferred, not a claim of deformable support. The original entry adds configurable object IDs, CLI demo budget/max trials and read-only diagnostics. Task algorithms and success thresholds are retained.

## Evidence and limits

- M2T2 `sage-original-m2t2-2f1632q1`: real SAGE rendering, actual CUDA PointNet forward/backward, 297 grasps, 3.17 s.
- cuRobo `sage-original-curobo-p6g6z7hm`: real GPU collision-aware planning, 31 steps, FK position error 4.94e-7 m, 5.47 s.
- Mug `sage-original-task-CsAC2Lpz`: two attempts, zero demos; diagnostic follow-up identified in `logs/phase3-original-task4-driver.log`, also zero demos. Robot reaches descent/lift targets, fingers close, but object stays on table. Contact/grasp suitability is the observed bottleneck; detailed grasp failure root cause remains unisolated.
- Cube `sage-original-task-LlLIdyMg`: one attempt, one demo, 68.84 s through simulator shutdown; sampled lift 4.65 cm, planar transfer 21.33 cm. Last sampled 1 s window has 11 samples and peak speed 0.000925 m/s; this is not a full-rate stability certification.
- All four actual task attempts took about 296.5 s through their process shutdown logs, plus short setup/probes/builds. Initial failed startup attempts are separately retained. No broad sampling campaign, policy training, real robot use or robot-phase GPT calls. GPU processes were cleared after completion.

Visual readback shows a large close-range surface in wrist RGB (possibly a debug frame marker); cause has not been isolated. Left/right views show the task clearly. Wrist camera image quality remains a separate prerequisite before policy training.

The successful HDF5 is 123 MiB on 5090-1. Local mirror keeps small reports, source patches, diagnostics, preview and 1.3 MB camera video. Sources/fixture/checkpoint fingerprints captured after the historical run are labeled as post-run evidence; they are not claimed as pre-run snapshots. Future launcher invocations capture their inputs before simulation.
