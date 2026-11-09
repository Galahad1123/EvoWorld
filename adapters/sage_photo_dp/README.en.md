# Photo-tabletop experiment adapters

Status (2026-11-08): native photo-scene collection20/20, batch32 /20,000updates /warmup1000 training completed. Validation selects epoch400. Actual online policy success8/10, expert replay2/2; full CPU trace/video audit passes. [Results, videos, scope and reproduction](../../experiments/phase8-photo-train-EEtcZBBs/ONLINE_EVAL.md). Earlier sections below retain implementation history; new-photo LeRobot conversion and RL remain pending.

## Collision bridge

`collision_bridge.bridge_planner_class(MotionPlanner, object_specs)` returns an opt-in subclass of the original SAGE MotionPlanner. Instantiate it with the original arguments. `object_specs` maps scene object names to `{'dynamic': bool}` and must match the actual RigidObject/AssetBase configuration. The existing unmodified collector does not automatically enable this bridge; the photo collector must use this factory, as demonstrated by `probe_collision.py`.

All enabled CollisionAPI Cube shapes are triangulated exactly, including invisible proxies. Child transforms/scales are baked. Parts are merged per scene object with correct vertex offsets, preserving original object names and grasp-target exclusion. Dynamic vertices remain in root coordinates for the original live PhysX pose path; static vertices are in world coordinates for the original AssetBase identity-pose path. Existing background meshes are preserved. Source USD, visual materials and simulator collision shapes are unchanged.

Unsupported enabled collider types fail explicitly. Exact triangle meshes without holes/left-handed orientation/approximations are accepted, but the current six-asset acceptance covers Cube geometry only. Dynamic root scale/shear is rejected. This bridge does not invent collision geometry for visual-only features such as the milk straw or monitor screen.

## Verified run

`experiments/phase8-collision-HUuKfh5F`:6 native assets,14 colliders,168 triangles,6 cuRobo meshes. Original cuRobo successfully planned a3cm end-effector lift with31 waypoints. Existing Phase7 Cartesian smoke executed160 control steps and passed independent CPU readback. The newly planned joint trajectory was saved but not executed; neither milk grasping nor new policy success is claimed. Ground is excluded in this tabletop planning probe, consistent with the existing planner setup.

Ten CPU tests pass:8 bridge tests and2 geometry tests, including invisible/disabled colliders, rotation/scaling, multiple parts, preserved background metadata, unsupported shape/polygon rejection and a real probe guard that raises on planning failure. Fresh review has no remaining Critical/Important findings for this scope. Initial launcher import-shadowing, relative experience-path, and original debug-export directory failures were retained as failed runs; the final run exits0 and has explicit success reports.

Run from EvoWorld root after sourcing `adapters/sage_robot/env.sh`:

```bash
RUN="$(mktemp -d experiments/phase8-collision-XXXXXXXX)"
envs/sage-robot/bin/python adapters/sage_photo_dp/probe_collision.py \
  --bundle experiments/phase7-integration-z8Yzxz8V/bundle --output "$RUN/probe" \
  --headless --enable_cameras \
  --experience "$PWD/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit"
envs/g1-realtosim/bin/python adapters/sage_realtosim/readback.py "$RUN/probe"
envs/g1-realtosim/bin/python -m unittest discover -s adapters/sage_photo_dp -p test_collision_bridge.py
envs/g1-realtosim/bin/python -m unittest discover -s adapters/sage_photo_dp -p test_geometry.py
```

Next: compose SAGE room with photo tabletop, bridge the M2T2 layout-mesh input, verify tabletop/wrist views and0.8x expert execution, then perform bounded milk-grasp pilot before collecting20 successful demonstrations. Collision compatibility alone does not clear those gates.


## Historical collection → training launch

`experiments/phase8-photo-train-EEtcZBBs` runs a frozen `pipeline.sh` in the background. Inspect `stage.txt`, `pipeline.log`, `collection.log`, `prepare.log`, `training.log`, and finally `pipeline-exit.json`. A missing exit file with a live PID means the pipeline is still running, not accepted as complete. The PID was897089 at launch; inspect command identity before acting on it.

Pilot `phase8-photo-pilot-9nWpEfpt` contributes2 independently verified successes in3 attempts. Bulk requests18 more, max34attempts. Original failed diagnostics count against the overall40attempt cap. Only after all20 exact saved intervals pass physical acceptance and image/action alignment, the job builds16/2/2 masks and starts batch32/20000optimizer updates/warmup1000. Every2000updates is checkpointed, and actual optimizer counts plus LR audits are saved. No deficient dataset silently trains.

Native policy views are tabletop+wrist128x128. Original left/right remain available for inspection. Actual poses/images are saved per attempt. Motion paths use0.8-speed interpolation; original stage machine retained, with tall-object lift target max(original placement height, sampled grasp height + required lift +4cm). Expert timeout500control steps is the original400steps divided by0.8. Onlineevaluation was pending at launch; the completed8/10 result is linked above.

The pilot used an earlier M2T2 material atlas and stale pre-override saved env cfg; native Isaac images were correct. Both differences are preserved in provenance. Corrected bulk saves cfg after native-scene override, uses texel-center atlas mapping, and adds tail angular velocity<0.2rad/s to physical acceptance. Derived HDF5 metadata comes from corrected bulk; no claim that old raw metadata was repaired. Source copies and original rejected runs remain preserved.
