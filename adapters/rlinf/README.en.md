# EvoWorld RLinf setup on 5090-1

2026-11-08: infrastructure smoke tests passed. This is an isolated RLinf/OpenPI Python environment sharing the **existing Isaac Sim 6.0.1 runtime**. No additional Isaac Sim distribution was installed. No pretrained policy weights, training job, EvoWorld RL task, reward design or success-rate evaluation was included.

## Use on 5090-1

```bash
cd ~/lizk/vlas/EvoWorld
source adapters/rlinf/env.sh
# Bounded runtime + one-environment simulator checks, with separate timestamped logs:
bash adapters/rlinf/check.sh
```

Use a separate shell for SAGE/DP and RLinf. The check script clears proxy variables only in its own process so Ray can communicate locally. Package downloads use Tsinghua; `HF_ENDPOINT=https://hf-mirror.com`. Git downloads used the existing local proxy. A small OpenPI tokenizer was obtained through the official Hugging Face endpoint after a mirror metadata failure; no model weights were downloaded.

## Components

| Component | Installed location / version |
|---|---|
| New Python environment | `envs/rlinf-py312`, Python 3.12.14 |
| RLinf | `upstream/RLinf`, `1bf78980d9339cbc67497eddceb9a1e3a74e2acc` |
| RLinf Isaac Lab fork | `upstream/IsaacLab-RLinf`, `4246b6b4f4a3e74ee20e002ed7536b1c788d39f4`, VERSION 2.3.0 |
| PyTorch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 |
| Ray | 2.52.1 |
| OpenPI helper package / NumPy | rlinf-openpi 0.1.1 / 1.26.0 |
| Existing shared simulation engine | `envs/isaacsim6/lib/python3.12/site-packages/isaacsim`, Sim 6.0.1 |

The new environment's `site-packages/isaacsim` and Isaac Lab's `_isaac_sim` are symlinks to that existing engine. The rest of the old environment's site-packages is **not** appended to the new interpreter's path. Do not install/uninstall Isaac Sim packages over these links. Kit may write its ordinary runtime caches; no old environment dependency or source was intentionally modified.

The first Python 3.11 trial environment (`envs/rlinf-openpi`) was removed after the working Python 3.12 setup passed. Its logs remain. Sim 5.1 was unsuitable for the current driver, consistent with the prior SAGE diagnosis. There was no driver change or global Ray shutdown.

## Compatibility changes

Saved patches apply only to the two new upstream checkouts:

- `patches/isaaclab-sim6.patch`: use the Sim 6 physics tensor API module; defer obsolete soft-body type annotations; guard absent legacy deformable schema and headless viewport; supply the original Isaac Lab 5.1 asset-library URL when the old cloud setting is absent. This does **not** validate deformable-body simulation.
- `patches/rlinf-sim6-launch.patch`: allow `EVOWORLD_ISAAC_EXPERIENCE` and `EVOWORLD_ISAAC_KIT_ARGS` in the stack-cube launcher. `env.sh` selects the existing Sim 6 base Python Kit experience and explicitly enables Lab cameras.

The original RLinf Isaac Lab headless experience returned empty camera buffers; the old TiledCamera copy kernel then accessed invalid GPU memory. Ordinary Camera and extra warmup did not fix that experience. Switching to the Sim 6 experience resolved the tested path. **Final task configuration retains the original TiledCamera, poses, task, reward and action processing.** Temporary camera/warmup experiments were reverted.

Sim 6 API changes are described in the [official Isaac Lab migration guide](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/isaaclab/migrating.html). This setup is a local compatibility adaptation of RLinf's pinned fork, not a claim of upstream certification for every task.

## Verification and limits

Authoritative records on 5090-1 are under `logs/rlinf-setup/` (small final records mirrored locally):

- `runtime-check.json`, `runtime-check-final.log`, `runtime-final-exit.txt`: independent interpreter, native OpenPI RL/config/Franka transform imports, official π0.5 PPO config composition, CUDA forward/backward with finite gradients, local Ray worker round-trip; exit 0.
- `sim-final/report.json`, `sim-final.log`, `sim-final-exit.txt`: official `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0` wrapper, one environment, reset, five zero Cartesian-action steps with open gripper, finite rewards/states, second reset, close completed; exit 0. Both RGB observations are `[1,256,256,3]`; state is `[1,8]`. Third-person and wrist PNGs were visually inspected.
- `existing-envs-verification.json`: all seven original environments retain identical package METADATA, `.pth` and `pyvenv.cfg` fingerprints (691 files total). This checks dependency/configuration integrity, not every runtime cache byte.
- `pip-check.txt`: **five declared version conflicts remain**: legacy rlinf-openpi pins Torch 2.7.1, and LeRobot 0.3.3 bounds Torch, torchvision, torchcodec and gymnasium below this simulator-compatible stack. Imports and the tested RLinf runtime work; this does not establish full model training, video-dataset loading or SFT compatibility. The existing LeRobot conversion environment is unaffected.
- The official wrapper emits a CUDA IPC tensor-lifetime warning during shutdown even after explicit close; the process exits 0. Multi-environment/multi-GPU training, rollout memory/performance, model forward/backward and long-running stability remain to be validated with the chosen task and checkpoint. FlashAttention and Apex were not installed.

These tests prove infrastructure and observation/action transport, **not** learned policy performance or a successful stack. No changes to the selected office scene or accepted v4 camera were made.

## Reproduction records

`requirements-frozen.txt` captures the final environment, including editable absolute checkout paths. It is a site-specific snapshot, not a portable solver-clean lock. Repository URLs: `https://github.com/RLinf/RLinf.git` and `https://github.com/RLinf/IsaacLab`.

Installation began with RLinf's official `requirements/install.sh` flow (`embodied --model openpi --env isaaclab --no-root --no-flash-attn --no-apex`). A local installer copy skipped its global Git-configuration cleanup. The final environment was rebuilt on Python 3.12 with Torch CUDA 13, and links to the existing simulator were created after dependency installation. Recreating it requires the pinned source checkouts, saved patches, this host's existing Sim 6 installation and deliberate handling of the documented legacy constraints; do not blindly rerun the original installer over the final environment. Exact installation attempts and intermediate pins remain in `logs/rlinf-setup/`.

Next: agree on the single-arm RL task, observations/action semantics, reset/success/reward and checkpoint; then validate actual model inference and a small rollout/update before increasing training resources.
