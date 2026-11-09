# g1-realtosim standalone installation on 5090-1

Phase 4 installation is accepted for the **CPU CLI/data utilities and USD bindings**. This is not acceptance of G1 GPU rendering, new reconstruction, robot control, LeRobot export or training.

Repository: `upstream/g1-realtosim`, pinned commit `eb80bd505a749b367d129a9283c672b8cc34317c`. No tracked source changes. Machine settings are in the Git-ignored `configs/local.evoworld.runtime.json`. Outputs are outside the repository under EvoWorld `experiments/`.

## Verified

Final evidence: `experiments/phase4-g1-install-a2pdg3cp/report.json` and individual command stdout/stderr. Earlier successful run `phase4-g1-install-d5bonsqe` is retained. The separate empty-test-suite negative harness is intentionally failed evidence, not an upstream test failure.

- Python 3.11.16 independent venv, editable package 0.1.0, seven locked dependencies; Tsinghua index used.
- All **15 upstream unittest tests pass**; dependency consistency passes.
- Installed console entry and module CLI help work. Doctor finds NumPy, h5py, pxr, Git, ffmpeg and ffprobe.
- Configured existing Isaac 6 interpreter exists and its installed distribution metadata is readable; no GPU startup was performed.
- USD bindings create and reopen a tiny metre-scale, Z-up in-memory probe. This is an installation fixture, not a reconstruction result or physical test.
- Original ingest command registers one image and two bounded video reference frames, verifies source SHA256 and leaves the reconstruction plan `awaiting_agent_reconstruction`.
- Ingest inputs are the existing SAGE simulation preview/video, explicitly **not real-world photographs**. No trajectory is presented as G1 data.
- Validator derives the actual test count and rejects empty test discovery. The negative harness simulates unittest reporting zero tests; its failure is expected.

Run again (CPU only, fresh evidence directory each time):

```bash
cd ~/lizk/vlas/EvoWorld
source adapters/g1_realtosim/env.sh
g1-realtosim --runtime upstream/g1-realtosim/configs/local.evoworld.runtime.json doctor
python adapters/g1_realtosim/run_validation.py
```

The installation validator deliberately pins the repository SHA and references the retained SAGE preview fixture. It is an installation regression script for this workspace, not a general G1 benchmark. All commands are bounded to 60 s; no API credentials are loaded.

## Recreate the CPU environment

On a matching machine / fresh project environment, first obtain the pinned repository and the already available Python 3.11.16 interpreter. Then:

```bash
source configs/mirrors.env
~/.local/bin/uv venv --python 3.11.16 envs/g1-realtosim
~/.local/bin/uv pip install --python envs/g1-realtosim/bin/python \
  -r adapters/g1_realtosim/requirements.lock
~/.local/bin/uv pip install --python envs/g1-realtosim/bin/python \
  --no-deps -e upstream/g1-realtosim
```

Exact packages: NumPy 2.4.6, h5py 3.16.0, Pillow 13.0.0, SciPy 1.17.1, imageio 2.38.1, imageio-ffmpeg 0.8.1, usd-core 26.11. This environment does not inherit SAGE or Isaac site-packages and does not install Torch.

ffmpeg/ffprobe are the Ubuntu 22.04 package `7:4.4.2-0ubuntu0.22.04.1`, downloaded from the existing Tsinghua apt mirror and **only extracted** under `tools/ffmpeg-jammy`. No apt installation or system configuration change was performed. Re-extraction from the retained `.deb`:

```bash
mkdir -p tools/ffmpeg-jammy
cd tools/ffmpeg-jammy
apt-get download ffmpeg=7:4.4.2-0ubuntu0.22.04.1
dpkg-deb -x ./ffmpeg_*.deb .
```

These executables depend on the preinstalled matching Ubuntu multimedia libraries; they are not standalone portable binaries. Exact package/binary SHA256 and `ldd` results are in `environment-manifest.json`; download log is `logs/phase4-ffmpeg-download.log`. Use the retained package if the mirror no longer hosts that exact version. `env.sh` changes PATH only in the invoking process.

Create `upstream/g1-realtosim/configs/local.evoworld.runtime.json` with `isaac_python` pointing to the existing `envs/isaacsim6/bin/python`; put the project ffmpeg directory into its child-process `env.PATH`. Do not copy historical SONIC paths or old driver-library overrides from another machine. Do not configure a nonexistent exporter interpreter.

## External dependency boundary

| Component | Current state / next use |
|---|---|
| Isaac Sim | Existing 6.0.1 interpreter configured separately; G1 renderer compatibility with Isaac 6 remains untested |
| Isaac Lab | SAGE's older adapted Lab is not assumed to satisfy G1 PT128 requirements; no G1-specific Lab environment configured |
| Blender | Not configured / not found in PATH; needed for later Blender reconstruction recipes |
| LeRobot + av | Optional `lerobot==0.4.3`, `av==15.1.0` not installed; use a separate compatible export environment later |
| Hunyuan3D | Model environment and weights not downloaded; needed only for that optional shape/texture route |
| SONIC/GRAIL | Online controller engineering, weights and robot assets remain external; integrations are reference snapshots |
| G1 raw trajectory + matching USD | Not provided on this host; required before a valid G1 saved-state render smoke test |
| Source photos / object assets | Needed for Phase 5 reconstruction; arbitrary photos have no guaranteed absolute scale |

Doctor checks module presence in its current CPU interpreter. Therefore `isaacsim=false`, `isaaclab=false`, `lerobot=false` there are expected and do not contradict the separately configured Isaac interpreter. Doctor does not start Isaac, validate its rendering, or certify the optional stack.

Follow repository `AGENTS.md`, `DATA_CONTRACT.md` and applicable skills before reconstruction/control/data work. G1's 43-joint plus two Revo2-hand contract is incompatible with the SAGE Franka 8-D action trajectory without an explicit future adapter. Existing replay restores saved states and does not prove new physics success. No source/asset publication, real robot operation, driver update, model inference or new GPU process occurred in Phase 4.
