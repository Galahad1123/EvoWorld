# Phase7: cached real-to-sim reconstruction in SAGE with Franka

Accepted experiment on5090-1: `experiments/phase7-integration-z8Yzxz8V`, successful runs `primary-fixed` and `reload`. The six accepted Phase5 photo-derived assets load through the original SAGE `FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser`/IsaacLab environment alongside the original cached Franka Panda. Original DLS action manager,12 policy observations and Phase6 RobotAdapter execute an actual short movement. M2T2,cuRobo and original SAGE task source remain unchanged; they are not invoked in this integration smoke.

This is the user-authorized **cached reconstruction path**, not new GPT inference from new photos. The same three user photographs and measured55×38×130mm carton reference, source evidence and final GPT program travel with the bundle. Robot placement is a simulation design because no real robot photograph/calibration was supplied.

## What was integrated

- `scene_spec.json`, `robot_spec.json`, `task_spec.json`: canonical v1 contracts for this fixed benchmark profile. See [SCHEMAS.md](SCHEMAS.md) and executable `specs.validate_specs`.
- `build.py`: copies six native generatedUSD assets/textures unchanged, packages original Panda with the original g1 generic packager, preserves photos/source hashes/provenance, and creates a composed `scene.usda` plus file manifest. No mesh conversion, asset scaling or primitive replacement.
- `runtime.py`: original SAGE Franka environment with native `RigidObjectCfg`(milk,tissue) and `AssetBaseCfg`(desk,tray,divider,monitor). It overrides robotUSD path with the bundled original asset, applies saved pose/joint defaults, retains original absolute8D hand-pose/gripper actions and12 policy observations, and reuses Phase6's live PhysX wrist-camera synchronization.
- `launch.sh`: starts a fresh bounded Isaac process, snapshots adapter sources before startup and fails on process status or missing explicit completion. `readback.py` independently checks actual states, transforms, action/observations, gravity settling, contacts and timing. `compare_runs.py` checks independent relocated initialization.

World_T_photo is translation[0,0,0.75]m: a **designed** table height. Franka base is[0,−0.48,0.751]m with yaw+90°, facing inward along+Y. No source asset changes size or relative layout. The desk is the original static partial tabletop; no reconstructed supporting legs are claimed. Floor and inspection cameras are designed simulation context.

## Actual verification

Each successful process ran160 control steps/320physics steps,3.2s simulated:100settle,30up and30return. Commanded vertical offset30mm; measured up displacement27.159mm, endpoint errors2.841/2.785mm. All nonbase robot contact sensors measured0N at recorded control steps. Both dynamic objects actually settled ~1mm under gravity, rather than being frozen at reset; final collision support errors about−3.00µm(milk) and−0.366µm(tissue). Last1s sampled speed peaks0.00234/0.000820m/s, position spans0 at float32 precision. State sampling is50Hz, physics100Hz; no unrecorded intermediate-contact guarantee is claimed.

The entire bundle was copied to `relocated/bundle`; a separate SAGE process used only that bundle's native assets/specs. Complete dependency closure and hashes passed, with Isaac built-in `OmniPBR.mdl` explicitly allowed by the original packager. Robot commands, joint/hand/base states and both object state arrays matched exactly between the two processes. Stochastic point ordering and pixel identity are not required. Actual runtime asset paths are saved, and all refer to the corresponding bundle.

19 integration CPU tests pass, including wrongunits/scaledframes/duplicateIDs/incorrectpose, missingtexture/changedrobotlayer, frozen or falling dynamicobject, wrongrobotbase, fake motion, unintendedcontact, stale wrist and incomplete trace. ExistingPhase6 tests19pass; originalg1tests15pass. Final independent review found no Critical/Important issue; the one Minor fixed-profile validation finding was resolved with regression tests. See `independent-review.json`.

Three GPU startup processes total: initial `primary` completed rendering/motion but failed report recording (old joint-limit field and NumPyfloat32 JSON). It is **not accepted** and remains preserved. `primary-fixed` repaired recording and passed; `reload` independently passed. No new API requests, weights, packages, training or real robot motion.

## Pictures and semantics

`reload/final-overview.png` shows the whole loaded robot/tabletop; `final-detail.png` shows objects and hand. `settled-*`, `raised-*`, `final-*` are still frames read from live sensors after real control steps, with software timestamps/intrinsics/extrinsics in matching `*-images.json`. They are not an offline replay or video-quality certification.12 policy fields are checked each step; full policy arrays are saved at these three snapshots. `trace.json` contains actual actions, robot/joint/object states, contact forces and camera poses/times. It is a diagnostic record, not an original SAGE Robomimic successful-grasp demonstration.

Existing physical/visual limits remain: carton folds/straw and tissue sheet have no separate collisions; paper is rigid. The monitor collision covers its base, not the screen. Zero measured contact refers only to authored collision proxies and this small motion, not complete scene obstacle modeling. Mass/friction and most dimensions are inferred/designed. Wrist visibility remains task dependent; no contact/grasp-quality vision acceptance. No approach/grasp/place, M2T2 candidate or cuRobo planning trial was run here.

## Reproduce on5090-1

From `~/lizk/vlas/EvoWorld`, existing environments only:

```bash
# CPU-only verification of existing bundle and actual recorded reload
envs/g1-realtosim/bin/python adapters/sage_realtosim/verify_bundle.py \
  experiments/phase7-integration-z8Yzxz8V/relocated/bundle
envs/g1-realtosim/bin/python adapters/sage_realtosim/readback.py \
  experiments/phase7-integration-z8Yzxz8V/reload

# Load the accepted portable bundle again into a NEW output directory
bash adapters/sage_realtosim/launch.sh \
  experiments/phase7-integration-z8Yzxz8V/relocated/bundle \
  experiments/phase7-manual-reload-NEW
envs/g1-realtosim/bin/python adapters/sage_realtosim/readback.py \
  experiments/phase7-manual-reload-NEW

# Rebuild from accepted Phase5 cache, run, relocate and independently reload
bash adapters/sage_realtosim/run.sh
```

The final `run.sh` is a convenience wrapper added after executing its constituent commands during acceptance; shell syntax was checked without spending another twoGPU starts. Both actual simulator launches used `launch.sh`. The source-only module import was subsequently qualified to avoid a unittest name collision; physics/runtime logic is unchanged and import checks pass in both environments. Pre-run execution snapshots and final validator/source hashes are retained separately.

`scene.usda` is a composed native asset scene for inspection. **Verified reload uses the SAGE loader and specs**, which apply configured joint reset and control. A bare genericUSD viewer can show the source articulation rest geometry; it does not run the initialized robot controller. Bundle portability requires the pinned SAGE/Isaac runtime and its built-inMDL library, not a new host installation.

Do not overwrite accepted evidence. No `.env` is loaded by any robot/build/check command. The source images remain private local project artifacts and are not committed or published. Stage8 manipulation baseline/RL is not started.

## Delivery

`integrated-scene-with-provenance.zip` contains the immutable portable bundle, input provenance, final adapters and selected successful evidence/preview images. Extract anywhere and use the existing SAGE project loader with the extracted bundle path; it is not an installer for a fresh host. Original execution snapshots remain separate from final validation-only/import hardening source snapshots. No extra GPU run was needed for those checks.
