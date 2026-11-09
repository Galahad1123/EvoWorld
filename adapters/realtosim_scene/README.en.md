# Phase 5: generic realtosim scene workflow

Accepted fixture on **5090-1**: `experiments/phase5-realtosim-lrrvvzrx`.
This is a controlled **synthetic image → approximate asset scene** workflow, not calibrated reconstruction of a real scene. G1 is absent. Future integration targets the original SAGE Franka/M2T2/cuRobo pipeline.

## What was reconstructed, reused, or designed

- Two 1024×1024 references are the left/right halves of frame 0 from the retained successful SAGE cube task video. `input-provenance.json` and original `ingested/inputs.json` retain source hashes. Original ingestion only prepares references; its initial `awaiting_agent_reconstruction` plan is intentionally preserved. The completed route is `route_plan.json` plus `scene.json`.
- One actual `gpt-6-astra` vision request, **low** effort, estimates a 5 cm red cube and approximate XY `[-0.23, 0.03]` in a new table-centered frame. Request/response/usage and uncertainty are retained. Axis signs are ambiguous; there is no source-camera pose calibration. Color is an approximate appearance, not measured albedo. Model-stated error ranges are not calibrated confidence intervals.
- The blue table uses unchanged official SAGE-10k `54a67a1f` mesh vertices, face UVs and PNG. Known asset dimensions supply scale; no new table mesh or texture was generated. Visual bounds are X ±0.692664981, Y ±0.398883283, Z 0.001–0.797781348 metres. Original metallic 0.3 / roughness 0.2 are reused.
- Table collisions are **five designed boxes** (top + legs). They support a tabletop drop, but do not reproduce the curved legs, trim, underside panel or detailed contact geometry. Do not use this result to certify robot clearance beneath/around the table.
- Cube mass 0.08 kg, static/dynamic friction 0.6/0.5, restitution 0, ground and support placement are **designed defaults**, not recovered physical measurements. Static table/ground are not dynamic mass estimates.
- Metres, Z-up, scalar-first `wxyz` quaternions. Stable IDs: `/World/{ground,table,block}`. The packaged scene has one dynamic body and seven collision prims. No robot, task controller, actions, rewards or training demonstrations are included.

## Results and data contract

Original unmodified `g1_realtosim ingest`, `build-scene` and `package-assets` are used. Independent reopening resolves two USD layers and one PNG entirely inside the relocated package; all references are relative. Mesh/UV values and PNG bytes match the source. A deleted-texture negative fixture is rejected by the original packager.

Actual Isaac6 PhysX integration produces **121 states / 120 steps / 2.000000104 s**. The validation-only initial position is 4 cm above the packaged placement (which already has a 1 mm gap). The cube falls 41.000 mm; final bottom-to-table gap is approximately −1.06 µm. Last 61 samples span a full second: peak speed 0.0002473 m/s, center span 4.53e-8 m. These are short, single-fixture numerical checks, not calibrated contact or long-horizon task results.

`physics/asset-physics.hdf5`, schema **evoworld.asset_physics.v1**:

| Field | Shape / interpretation |
|---|---|
| `time` | 121 monotonic relative seconds, 60 Hz |
| `block/pose_world` | 121×7: XYZ metres then quaternion wxyz |
| `block/{linear,angular}_velocity_world` | 121×3, m/s and rad/s |
| `images/rgb` | 4×600×800×3 uint8 |
| `images/state_index` | `[0,120,120,120]`, references the recorded pose |
| attrs `cameras`, `timing`, `image_semantics` | authored USD cameras, measured step/callback timing, explicit offline-image semantics |

**Images are offline frozen-state rerenders after physics stops**, not synchronous sensor samples. The initial/front, final/front, final/back and final/front/lower-light views are generated from recorded poses with the rigid body disabled and Replicator `delta_time=0`. `integrated-states.npz` is written before rendering; the CPU reader requires it to match HDF5 values exactly. It checks USD displayed poses and red-cube image projection (maximum error 0.65 px), PNG/HDF pixel equality and the visible initial elevation. Camera extrinsics use USD local camera axes (+X right, +Y up, viewing −Z) and column-vector matrices; these are authored cameras, not real camera calibration. No sensor-latency or video-quality claim.

The current accepted `physics/` reuses the unchanged integration from `physics-preview-failed-5/` for a preview-only recovery. `reused-integration-source.json` explicitly records that dependency. No new physical rollout was performed in that final preview process. `execution.json` wall time describes that process, not cumulative experiment time.

## Reproduce without another API request

Run from the remote project root. Create a fresh output; do not overwrite accepted evidence. The cached source NPZ is an editable mesh/UV conversion retained alongside its original PLY hash.

```bash
cd ~/lizk/vlas/EvoWorld
source adapters/g1_realtosim/env.sh
phase5_source=experiments/phase5-realtosim-lrrvvzrx
phase5_new=$(mktemp -d experiments/phase5-rebuild-XXXXXXXX)
cp -a "$phase5_source/input-views" "$phase5_source/source-assets" "$phase5_new/"
cp "$phase5_source/input-provenance.json" "$phase5_source/image-estimate.json" "$phase5_new/"
python -m g1_realtosim ingest "$phase5_new/input-views" "$phase5_new/ingested"
python adapters/realtosim_scene/build.py "$phase5_new"
python adapters/realtosim_scene/check_package.py "$phase5_new"
```

The above is a **cached-estimate rebuild**, not a new GPT reconstruction. For a genuinely new request after preparing suitable references, `estimate.py` runs once with timeout90/retries0/max2048 tokens through the credential wrapper:

```bash
envs/sage/bin/python scripts/with_gpt_env.py -- envs/sage/bin/python adapters/realtosim_scene/estimate.py "$phase5_new"
```

The estimate prompt is specific to this known synthetic table fixture. Real inputs require an appropriate scale anchor, uncertainty and coordinate/route review; swapping image filenames alone does not establish calibrated reconstruction. No Qwen or other model weights are needed here. Network/mirror settings remain scoped to the existing wrapper/configs.

Optional new bounded physics run and CPU verification, with no credentials in the process:

```bash
source configs/mirrors.env
timeout --signal=TERM --kill-after=15 90 envs/isaacsim6/bin/python -u adapters/realtosim_scene/physics.py "$phase5_new" > "$phase5_new/physics-process.log" 2>&1
cp "$phase5_new/physics-process.log" "$phase5_new/physics/process.log"
envs/g1-realtosim/bin/python adapters/realtosim_scene/review.py "$phase5_new"
```

For preview-only recovery, pass an existing directory containing `integrated-states.npz` and `integration.json` as the second argument to `physics.py`, after retaining the failed `physics/` directory under a new name. The recovery must use the same scene fixture. Successful stdout marker **and** independent reader success are required; process exit alone is insufficient. Failures now pass a nonzero `exit_code` to `SimulationApp.close`, including fast shutdown. The final full script's integration and preview paths were verified in separate processes during recovery; no extra end-to-end GPU rerun was spent after the fix.

## Failures, budget and portability

Five earlier attempts remain under `physics-failed-1`, `physics-image-mismatch-2`, `physics-failed-3`, `physics-failed-4`, `physics-preview-failed-5`, each with source/logs. Attempt2's preliminary passing readback is **superseded**: its initial image showed the wrong pose. Causes: Replicator returned a camera parent; old `World.reset` internally steps then restores PhysX state without matching USD; writing render poses during active dynamics can resynchronize state; stopped-timeline annotations require explicit Replicator capture. The final workflow separates integration from frozen rerenders and checks projection. No successful result was discarded or relabeled silently.

Initial budget was two GPU launches; correctness debugging used **six short Isaac processes**, final one preview-only, approximately six seconds each (startup/shutdown excluded from some per-process timings). One scene group, one GPT request (2761 input /467 output/3228 total tokens, 14.78 s), no model downloads or training. No peak GPU memory was sampled. All task GPU processes exited; the pre-existing desktop remains.

`portable-scene.zip` contains runtime USD dependencies only. **Use `scene-with-provenance.zip` for handoff**: it also includes source labels, uncertainty, editable scene specification, reports and this README. Historical source paths in the manifest are provenance, not required dependencies. Full editable source mesh arrays and input views remain in the experiment directory. Neither archive contains credentials. Upstream tests: 15/15; reader negative cases reject invalid timestamps and static trajectories.

Reusable modules: ingestion/reference manifests, route/spec discipline, generic USD/material/collision construction, asset packaging/dependency checks. G1-specific trajectory render/control/export code, G1 joints/root/control labels and locomotion supervisors are **not** carried into Franka. This asset HDF5 is also not the original G1 HDF5 contract or SAGE 8-D robot-action schema. Phase6 should audit transforms, fixed-base Franka placement, camera/data schemas and collision interfaces before Phase7 integration.
