# SAGE native demonstrations → Diffusion Policy → LeRobot

All execution is on 5090-1. This is Phase8A imitation-learning infrastructure;
it is not VLA training or RL acceptance. Current run evidence is under
`experiments/phase8-dp-lk1kG3Hz`, with a second collection batch recorded there.
**Pipeline verified; learned task NOT successful.**20 successful expert demos/26 attempts,4163 transitions;16/2/2 split. Final DP trained3000 updates in115.59s (plus an earlier1000-update/37.09s pilot), with2.22GB peak allocatedGPU memory. Expert replay2/2, learned independent test0/2. Final train/validation noise MSE0.02527/0.02746. LeRobot20/4163 numeric and boundary round-trip passes;13 tests and independent review pass. See `acceptance.json`, `training-curve.png`, and `policy-test-3000/report.json`.

## Collection

The original M2T2, cuRobo, state machine and RobomimicDataCollector run on the
Phase3 procedural cube fixture in the existing SAGE room. No new GPT calls.
`collect.py` records its exact executed source and adds seeds, runtime camera
compatibility and read-only physics/initialization evidence. The Phase6 wrist
camera fix and disabled debug markers are applied. Later batches use320x192
auxiliary diagnostic cameras; original policy cameras remain128x128.

```bash
EVOWORLD_NUM_DEMOS=20 EVOWORLD_MAX_TRIALS=32 \
EVOWORLD_COLLECTION_TIMEOUT=1800 EVOWORLD_DP_SEED=82 \
bash adapters/sage_dp/collect.sh
```

Every launch writes a fresh experiment. The shell launcher freezes itself
because modifying a running Bash script can corrupt subsequent reads.
The first historical batch completed8 demonstrations/10 attempts, but its
launcher was synchronized while running and failed after the simulator returned.
`launcher-exception.json` preserves wrapper exit2 and UNKNOWN simulator exit;
it is accepted only with the original completion marker, independent HDF5
readback and action-aligned physical verification. No exit code is invented.

## Dataset and training

`prepare.py RUN --extra-run SECOND` merges native episode groups into a derived
HDF5, preserving raw files. Exactly20 successes are expected by default;
16 train/2 validation/2 test are split by whole episode with seed108. Repeated
action trajectories, bad quaternion/gripper values, temporal inconsistencies,
nonfinite values and incomplete collection are rejected. Failed construction
leaves a distinctly named incomplete file, never a valid training filename.

`verify_experts.py RUN` binds every saved action to the400-step physical trace,
and checks the actual saved interval for lift, placement, release, support and
settling. Source state snapshots and raw-data hashes preserve provenance.

```bash
envs/g1-realtosim/bin/python adapters/sage_dp/prepare.py RUN --extra-run SECOND
envs/g1-realtosim/bin/python adapters/sage_dp/verify_experts.py RUN
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
envs/sage-dp/bin/python adapters/sage_dp/train.py RUN
```

Training uses SAGE's bundled robomimic0.5 DiffusionPolicyUNet, RGB left/right,
ee_pose and joint_qpos. The initial bounded budget is20×50=1000 optimizer steps,
batch16, AdamW1e-4, EMA, DDIM10 inference steps/100 training diffusion timesteps,
prediction horizon16, action horizon8. Visual backbones start without pretrained
weights. Wrist/depth/points stay in native raw data but are not initial policy
inputs. Training-derived min/max action statistics are shared with validation;
the original fork otherwise computes separate scales, making losses inconsistent.
The final wrapper also prevents validation from advancing the LR scheduler: the original fork calls `on_gradient_step()` even in validation. Audits assert unchanged scheduler state and sampled weights. Missing `train.end_epoch` is supplied explicitly; protobuf4.25.8 fixes TensorBoard import in the overlay. Validation does not supply gradient updates. Training seed108 is explicitly set
because the bundled entry point comments out its own seed-setting lines.

The final accepted training command adds `--name dp_rgb_3000 --epochs 60`. It is a fresh3000-update run, not a resume of the1000-update pilot. No further training was launched after independent testing. The pilot used the fork's eager, unused CLIP import from existing cache; final training/inference use `bootstrap.py` to bypass language initialization and reject any language-conditioned request. No CLIP download is required for this RGB/state policy.

`sage-dp` is a Python3.12 overlay that reuses existing IsaacSim Torch2.11/cu130
and torchvision0.26 through a .pth. Its old diffusers0.11.1, Hub0.23.4 and
transformers4.41.2 live only in the overlay. S3 support is an import dependency
of the original fork; local data does not require AWS credentials.

## Simulator evaluation

`rollout.py` loads the same native scene and restores a held-out episode's
initial physical state and cameras. Twelve render-only warmup frames plus explicit sensor refresh prevent black/stale initial images, without advancing physics or applying expert actions. Accepted initialRGB MAE is2–3/255. Robot observation restore is checked within
1e-3; initial images and all actions/object states are saved. Expert mode replays
recorded commands as a control. Policy mode never uses expert future actions:
`policy_worker.py` loads the checkpoint in sage-dp and communicates over private
local pipes. This avoids inserting training dependencies into the simulator.

Commands use the original base-frame absolute8D pose/gripper action. Predicted
quaternions are normalized and gripper outputs thresholded at0; raw outputs are
also retained. Evaluation runs50Hz, saves25FPS video, and adds50 steps holding
the final command. Physical success requires lift>3cm, final XY error<8cm,
finger mean opening>2.5cm, final object Z within1.5cm of support and last0.5s
speed<5cm/s. All height comparisons use COM coordinates, not the cube's bottom-origin USD root. An initial rejected verification report mixed these by2.5cm and is retained as `expert-physical-verification-origin-bug.json`. This is a fixed-scene, small-sample result, not broad generalization.

## LeRobot isolation

The independent CPU-only `envs/sage-lerobot` pins LeRobot0.4.3, Torch2.7.1+cpu,
torchvision0.22.1+cpu and torchcodec0.4.0; dependencies come from Tsinghua,
CPU Torch wheels from the official PyTorch CPU index. Its dependency check passes.
No .pth links to SAGE or DP, no GPU use and no Hub upload.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 \
envs/sage-lerobot/bin/python adapters/sage_dp/export_lerobot.py \
  --hdf5 RUN/training.hdf5 --output RUN/lerobot-v3

# Reload from disk in a fresh process:
envs/sage-lerobot/bin/python adapters/sage_dp/export_lerobot.py \
  --output RUN/lerobot-v3 --verify-only
```

The official writer produces LeRobotDataset v3.0. All12 current observation
fields, actions, rewards and original done flags are exported. RGB uses H264;
depth/points and other numerical arrays are exact. `next_obs` stays in HDF5,
including final next observations. Do not describe the export as lossless RGB
or a complete RL transition dataset.180 sampledRGB frames were checked, maximum
MAE7.07/255; every exported numerical value and20 episode-boundary action chunks
were checked. Native HDF5 and source hashes remain unchanged.

**Split selection is explicit.** Official `meta/info.json` defaults to all20
as train. Use `train-dataset-config.json`, `valid-dataset-config.json` or
`test-dataset-config.json` as the `dataset` portion of a downstream training
configuration. These carry the required `episodes` list. `split-indices.json`
and actual LeRobot reader checks prove disjoint16/2/2 membership. Loading only
repo_id/root without episode selection loads all20. Pin the target VLA
repository's compatible LeRobot version; v3 does not imply compatibility with
older v2-only loaders.

```bash
HF_HUB_OFFLINE=1 envs/sage-lerobot/bin/python -m unittest discover \
  -s adapters/sage_dp -p 'test_*.py' -v
```

All13 tests pass, including real official writer/reader video roundtrip,
malformed/duplicate/temporally inconsistent data, success metrics, scheduler
validation behavior and explicit split isolation. Actual GPU jobs have exited.
The automatic frozen-launch prologue was added after the collection runs and
is shell/prologue tested; the second accepted batch used a manually frozen
launcher. Training does not establish task competence: both independent policy
rollouts failed to lift or transfer the cube. Reward/termination and RL training,
Phase7 photo-scene manipulation, and broader generalization remain outstanding.

## Recorded-observation open-loop evaluation (2026-11-08)

Evidence: `experiments/phase8-dp-lk1kG3Hz/open-loop-20261108-112743/`.
The unchanged final checkpoint is evaluated on `demo_10` (204 frames) and
`demo_7` (196 frames), with seeds208/209/210. These are400 unique expert frames,
not1200 independent samples. Each mode scores1200 predictions across repeats.
No simulator, training, model selection, or new dependencies are involved.

| Method | Mean position error (cm) | Position p95 (cm) | Mean orientation error (deg) | Gripper accuracy |
|---|---:|---:|---:|---:|
| Original8-step queue | 2.109 | 5.738 | 6.797 | 91.33% |
| Fresh prediction each frame, first action only | 1.955 | 5.069 | 8.169 | 97.50% |
| Hold observed ee_pose / previous gripper command | 2.919 | 6.320 | 3.809 | 97.50% |

`chunk8` uses the same official RolloutPolicy preprocessing, EMA, DDIM10,
denormalization and queue as closed-loop deployment; `fresh1` resets the queue
per frame. Both read only recorded obs, predict against actions at that same
frame, and discard unfinished chunks at episode boundaries. No padding or
expert action is fed into the model. The generated16-step latent horizon is
not fully scored: only the executable8-step chunk or its first action is used.

Position is base-frame target XYZ error. Orientation is q/-q invariant
geodesic error after quaternion normalization, matching deployment. Raw predicted
quaternion norms reach0.13–0.17; this is not evidence of unit-quaternion output.
Gripper uses the same >=0 threshold as deployment. High fresh1 gripper accuracy
alone is weak evidence, since simply preserving the observed previous command
also reaches97.5%. This comparison does not prove shorter action chunks would
fix closed-loop execution: position/gripper improve slightly but orientation
worsens, on only two trajectories. The original closed-loop result remains0/2.

Artifacts include raw predictions/reference actions (`*-predictions.npz`),
per-frame CSV, per-episode/seed/queue-offset metrics, command-switch windows,
and two plots. Plots show seed208 lines and the three-seed position envelope;
they are action comparisons along the expert timeline, not robot rollouts.
Closed-command subsets are fully summarized; open-command recall/count is
reported, with raw predictions available for further subsets.

```bash
RUN=experiments/phase8-dp-lk1kG3Hz
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
envs/sage-dp/bin/python adapters/sage_dp/evaluate_open_loop.py \
  --run "$RUN" --checkpoint "$(cat "$RUN/final-checkpoint.txt")" \
  --output "$RUN/open-loop-$(date +%Y%m%d-%H%M%S)"
envs/g1-realtosim/bin/python adapters/sage_dp/verify_open_loop.py \
  "$RUN/open-loop-20261108-112743"
```

Actual evaluation took34.80s including setup and plots, with2.215GB peak
PyTorch allocation. GPU returned to desktop baseline afterward. All18 adapter
tests passed (5 new). CPU verification binds references to original HDF5,
checks immutable checkpoint/data/source hashes, independently recomputes primary
position/orientation/gripper metrics, and checks CSV identities/timing/counts.
It does not claim recomputation of every JSON subset or CSV numerical field.
Fresh review independently checked quaternion results via SciPy rotations;
no Critical/Important findings. No new videos were generated because no actions
were executed in simulation; the previous closed-loop videos remain available.
