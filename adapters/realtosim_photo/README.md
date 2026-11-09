# Phase 5 real-photo reconstruction acceptance

Accepted on 5090-1: `experiments/phase5-photo-f92ngyB1`, selected `round3/build/relocated/scene.usda`. This supersedes the earlier synthetic cube fixture as Phase5 acceptance evidence. Three user photos and the measured carton reference 55 × 38 × 130 mm drove actual GPT-authored source, two feedback/patch rounds, original g1-realtosim ingest/build/package, relocated USD reload, Isaac rendering and independent physical readback.

## Evidence

- Three successful `gpt-6-astra` medium calls: initial program, four source patches, nine source patches. Exact API programs/patches match executed source hashes. Five API attempts total: two earlier high calls failed; their usage is unknown. Known completed usage is 77,831 tokens. No new weights or training.
- Six new assets: printed folded carton and hollow straw, rounded tissue wrapper and wrinkled paper, desktop with real opening, cable tray, divider and monitor. Ten textures (nine photo-derived, one procedural). No SAGE table reuse in this run.
- All packaged USD/texture dependencies relative and contained. Missing texture and doubled scale rejected. Seven adapter tests and fifteen original tests pass.
- Three static render processes, one physical process. Physics: 120 steps at 60Hz, 121 states, ~2s, milk/tissue drop ~21mm. Last-second peak speeds 0.00434/0.000328m/s, final support errors −4.32/−1.11µm. Full traces independently checked.
- `round3/build/preview/*-comparison.png`: original left / Isaac right. `overview.png` and lower-light view inspect geometry/materials. These are nominal assembly static renders, not synchronized physical observations.
- `physics/states.hdf5`: `evoworld.photo_asset_physics.v1`, time + milk/tissue poses (XYZ,wxyz), linear/angular velocity; no actions/rewards/images or robot demonstration.
- Frozen cameras use assumed shared focal1450px, center principal point, zero distortion. All landmarks are fitting data. Final rounded-geometry milk/tissue RMS px: main13.61/11.40, left1.64/8.96, right1.55/3.37. No independent metric calibration claim.

## Limits and provenance

55×38×130mm is the measured reference envelope; 66mm opened-ear width, straw extension, body/roof split and all other sizes are inferred. Milk/tissue masses0.235/0.035kg and friction0.5/0.4 are design defaults. Fold ears, straw and paper are visual geometry without separate collisions; paper is rigid, not cloth. Wrapper text stretching, overly planar paper and approximate monitor/tray/background alignment remain. Stability does not prove graspability. Final GPT feedback saw round2, then authored round3; actual final output was accepted by rendering, physics and fresh independent review.

Minor reviewer finding retained: scale checker covers measured footprint and broad overall height; current source's carton ears reach130mm, but dedicated carton-only Z-scale negative regression is future work.

## Repeat on 5090-1

From `~/lizk/vlas/EvoWorld`; no credentials required for cached program/CPU verification:

```bash
envs/g1-realtosim/bin/python -m unittest discover -s adapters/realtosim_photo -p 'test_*.py' -v
envs/g1-realtosim/bin/python -m unittest discover -s upstream/g1-realtosim/tests -v
envs/g1-realtosim/bin/python adapters/realtosim_photo/check_assets.py experiments/phase5-photo-f92ngyB1/round3/build
envs/g1-realtosim/bin/python adapters/realtosim_photo/readback.py experiments/phase5-photo-f92ngyB1/round3/build
envs/g1-realtosim/bin/python adapters/realtosim_photo/final_residuals.py experiments/phase5-photo-f92ngyB1/round3/build
```

Original build commands and generated-source hashes are retained per round in `tool-execution.json`; prompts, response, route plan, source code, exact patches, usage and failed-attempt logs remain. The accepted build is immutable evidence; rebuild into a fresh directory. `scene-with-provenance.zip` contains portable scene, photos, source/feedback/provenance and validation. No credentials are included. The source-generated metadata's “awaits validation” status is historical; `acceptance.json` is the post-validation decision.
