# SAGE office background option

**Current user selection: candidate05, `layout_acb3ece0` (Contemporary office).** Preferred background and verified preview are recorded in `configs/selected-office.json`. The initial retro-office experiment below remains historical evidence.

2026-11-08: implemented and rendered on5090-1. This is an explicit background option for the accepted native photo-tabletop/Franka task, with no default change to historical collection/training/evaluation scripts.

The user requested an existing SAGE office in place of the green/brown floor and dark backdrop. We selected official SAGE-10k `layout_a5ac58b7`, room type `office`, style `Retro 1950s Diner-style office`. This is a **retro office**, not a claim of a conventional modern office. The original archive and official previews remain preserved. Source: `https://huggingface.co/datasets/nvidia/SAGE-10k/blob/main/scenes/20251213_170153_layout_a5ac58b7.zip`; download used hf-mirror via5090-1's existing local proxy, no credentials. Archive75,091,146bytes, SHA256 `3eefa25730194afca3186212a0dde1c139e828f62cf53c22af08efd9a2abf584`. Metadata-only range search examined36scene archives; only one full archive was downloaded. No dependencies installed or GPT requests made.

## Composition

- Original room5×4×2.7m, translated by[−0.42,−0.03,0]m to surround the unchanged task coordinates.
- Reuse original office meshes/textures. Remove the ceiling for open-top inspection/lighting, and three stools plus one printer whose actual mesh bounds overlap the reserved task workspace XY[1.10,1.00]–[3.00,2.85]m. Descendant removal follows support relationships.
-68retained meshes, packaged in `assets/SAGE-10k/office-a5ac58b7-background-v1/office.usdz`; SHA256 `a87970c01d7f73a7b55a448a898153658f9561545200caa3d5ead38de9d6e07f`. Static triangle-mesh collision; peripheral furniture/doors are frozen as background. All package dependencies resolve.
- Remove previous room/floor/wall/door/window entries. Preserve the six GPT assets, Franka, task coordinates, original tabletop/wrist camera intrinsics/extrinsics and128×128policy resolution. White dome intensity1500 and a separate960×720overview camera are rendering choices for this variant.
- `runtime.configure(cfg, background)` runs after `photo_runtime.configure`; `runtime.attach(env, settings)` places the overview camera. There is no implicit environment switch for the frozen20k model or previous8/10 evaluation.

## Evidence

`experiments/phase8-office-preview-20261108-142157/`:

-[Room and robot overview](../../experiments/phase8-office-preview-20261108-142157/office_overview.png)
-[Tabletop camera](../../experiments/phase8-office-preview-20261108-142157/tabletop.png), [wrist camera](../../experiments/phase8-office-preview-20261108-142157/wrist.png)
-[Report](../../experiments/phase8-office-preview-20261108-142157/report.json), [independent CPU audit](../../experiments/phase8-office-preview-20261108-142157/verification.json)

Renderer and driver exit0.100control steps/2simulated seconds holding the restored robot pose; milk maximum displacement4.01µm, final0.5s maximum speed0.00266m/s. Source snapshots, native-bundle hashes, saved stage and raw trace retained. Full stage/USD assets stay on5090-1; small artifacts mirrored locally. The original photo-only `scene-profile.json` initially retained the old room label; the report and frozen source already showed the correct office. A subsequent provenance-only annotation corrected that label and retained the original file. The current preview script writes the office provenance directly.

This establishes successful loading/rendering and a short hold smoke check. It does **not** establish office-scene expert planning, policy success or retraining. The earlier8/10 result remains scoped to the old background. Changing RGB appearance requires a separate new evaluation and, for training on this scene, new/re-rendered visual demonstrations.

## Reproduce

From EvoWorld root on5090-1; use fresh output directories. No existing dataset or checkpoint is overwritten.

```bash
source adapters/sage_robot/env.sh
# Build once; output must not exist:
envs/sage-robot/bin/python adapters/sage_office/build_background.py \
  --layout assets/SAGE-10k/office-a5ac58b7/layout_a5ac58b7.json \
  --output assets/SAGE-10k/office-a5ac58b7-background-NEW
# The accepted packaged background can be reused directly:
envs/sage-robot/bin/python adapters/sage_office/preview.py \
  --background "$PWD/assets/SAGE-10k/office-a5ac58b7-background-v1" \
  --output "$PWD/experiments/phase8-office-preview-NEW" \
  --headless --enable_cameras \
  --experience "$PWD/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit"
```

Build was bounded at180s, preview at240s; actual app runtimes about12.4s and15.9s respectively. GPU processes exited. No real hardware involved.

## Offline office selection gallery (2026-11-08)

Completed user-approved gallery:15general office candidates,30official previews, stable numbers01–15, Chinese style tags plus original descriptions, room dimensions/object counts, two-view switching/full-size links, search, selection persistence and copied/exported IDs. Excluded two dentist-office candidates. Current composed retro background is01;09is Scandinavian and05contemporary. Style descriptions are metadata, not visual-quality guarantees.

Canonical portable output on both hosts: `galleries/office-20261108/index.html`; ZIP `galleries/office-20261108.zip` (73,519,256bytes, SHA256 `7eae90d6d599b8ea89a15cb08f8829abc522344ccb306f68b8303021539e991e`). Double-click the HTML or unzip the package and open its index; no server or dependency installation is needed. Report selections by candidate number/layout ID. Gallery does not send selections to a server.

Collection run:`experiments/sage-office-gallery-20261108-143712`, pinned official dataset revision `d425de49ff445fd3903d790a11461d23ffd0c7dd`. Reused36metadata records/two cached previews; inspected330additional archives through ZIP-prefix/CRC-validated HTTP ranges usinghf-mirror and six bounded workers,289.87s,96,455,784bytes transferred. Found17office-labelled scenes including two dental rooms;15remain after filtering. No newfullscene archives, modelweights,GPTcalls,training or simulation. Asset archives remain the same two downloaded previously.

Commands from the5090-1projectroot:

```bash
envs/sage/bin/python adapters/sage_office/collect_gallery.py \
  --output experiments/NEW-GALLERY-RUN --target 16 --workers 6 --max-scan 1200
envs/sage/bin/python adapters/sage_office/build_gallery.py \
  experiments/NEW-GALLERY-RUN --output experiments/NEW-GALLERY-RUN/gallery
```

`collection.json` retains every result/error and transfer counts; `index.json` pins remote archive paths/sizes. Original preview bytes are copied unchanged; independent local verification checks all30SHA256values, unique layout IDs/contiguous numbering andZIP CRC. Isolated localFirefox tests pass for all30images, view/full-size link updates, Chinese search, selected-only/empty states, reload persistence, export payload, clipboard-denied manual fallback, clearing and narrow-screen layout. Screenshots and`browser-verification.json` are inside the gallery. `galleries/office-20261108-verification.json` records the final package hash. Browser session/driver exited.

The first generated HTML had a JavaScript newline escaping error caught by actual browser interaction andNode syntax checking; the raw template fixes it. Historical first/final/release build directories remain in the collection run. The canonical gallery additionally clarifies that no *new* full assets were downloaded in this round. This is a visual selection tool, not scene/physics acceptance. Next action: use the user's chosen ID to download only that scene and prepare a composed preview.

## User-selected candidate05: layout_acb3ece0

The user selected gallery05. Downloaded only its114,066,656byte archive throughhf-mirror at pinned dataset revision`d425de49ff445fd3903d790a11461d23ffd0c7dd`. ArchiveSHA256`90b6acefc6336ef13dd6919f922110d354fc02f79dd373ca2e353ecb4a7e4187` matches officialLFS metadata;ZIP CRC and layout equality against gallery metadata pass. Source:`scenes/20251215_014646_layout_acb3ece0.zip`.

Original4.5×5×2.7m room translated[−0.17,−0.53,0]m around the unchanged task. Retain96static meshes; remove29original objects in the reserved robot/table workspace plus ceiling. NativeGPT bundle hashes, object specs and task shift match the prior office variant. Backgroundpackage:`assets/SAGE-10k/office-acb3ece0-background-v1/office.usdz`,SHA256`c5dcc21e22f17f63eb9db240ec8dbe8b2f5476f0c73ef690e9b1531cebef5a39`,zero unresolved dependencies.

Verified run:`experiments/phase8-office05-preview-20261108-151115`. Build and renderer exit0;100steps/2s hold,maxmilk displacement4.01µm,last0.5s maximum speed0.00266m/s. [Composed overview](../../experiments/phase8-office05-preview-20261108-151115/office_overview.png),[tabletop](../../experiments/phase8-office05-preview-20261108-151115/tabletop.png),[wrist](../../experiments/phase8-office05-preview-20261108-151115/wrist.png),[independent audit](../../experiments/phase8-office05-preview-20261108-151115/verification.json). Small artifacts mirrored locally;fullassets/stage remain5090-1.

Exact commands after sourcing`adapters/sage_robot/env.sh` (existing output paths must not be overwritten):

```bash
envs/sage-robot/bin/python adapters/sage_office/build_background.py \
  --layout assets/SAGE-10k/office-acb3ece0/layout_acb3ece0.json \
  --output assets/SAGE-10k/office-acb3ece0-background-v1
envs/sage-robot/bin/python adapters/sage_office/preview.py \
  --background "$PWD/assets/SAGE-10k/office-acb3ece0-background-v1" \
  --output "$PWD/experiments/phase8-office05-preview-20261108-151115" \
  --headless --enable_cameras \
  --experience "$PWD/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit"
```

`configs/selected-office.json` records05as the preferred backdrop for subsequent photo-scene work;`logs/office-preview-current.txt` points to this preview. Frozen gallery numbering and historical training/evaluation remain unchanged (gallery01's“current”badge describes its creation-time state). No new policy evaluation, expert planning or retraining is implied. GPU processes exited;noGPTcalls,packages or hardware operations.

## Full-room presentation views

User clarified that the tabletop close-up did not show the complete room. `preview.py --full-room` adds an independent1600×1200oblique cutaway and a four-wall topdown image, leaving default task/camera configuration unchanged. Canonical output:`experiments/phase8-office05-fullroom-20261108-152846/room_topdown.png`;`office_overview.png`is the oblique cutaway. Two near-side wall meshes are hidden only for the cutaway and restored before topdown/savedstage. No asset/physics geometry edits. Nativebundlehashes match the selected05preview;exit-status/report/driver success and image dimensions/readback verified. The first attempt`phase8-office05-fullroom-20261108-152736`failed because wall-name matching also selected material prims; fixed by requiringMesh type. Its simulator shell exit0 did not override recordedexit-status1;failure retained. No training/policy evaluation.

Floor appearance diagnosis for05: original archive/extracted/packaged`room_efdc4606_floor.png`are byte-identical(SHA256`f6f43967d19e50a106e38ce1b8a9c340003f4b85a9ed40d8f117d44ebaa802cf`).RGBmean0.06246/255;the official source texture is nearly black. Valid material binding and package dependency paths;no new material edit. See fullroom run`floor-material-diagnosis.json`. Original artistic intent is not inferred.

### 2026-11-08: User-approved light-gray matte floor for office05

Created reversible background `assets/SAGE-10k/office-acb3ece0-background-v2-gray/office.usdz`. Only floor diffuseColor changed to linear RGB [0.55,0.55,0.55]; roughness 1 and metallic 0 retained. USD audit passes: 96 meshes, 5570 other attributes, relationships, collision geometry and packaged texture bytes unchanged; zero unresolved dependencies. Original package retained.

Run `phase8-office05-gray-20261108-153821` renders 1600x1200 full-room views successfully. Native GPT bundle hashes unchanged; 100-step/2-second hold traces match the original floor exactly, milk displacement 4.01 micrometers. Walls restored, render exit 0, CPU verification saved. Selected-office config and preview pointer now select this gray-floor variant. No training/policy evaluation; simulator GPU process exited. Next work remains subsequent task validation in selected office05.

Reproduction (from project root; use fresh output paths):

```bash
envs/g1-realtosim/bin/python adapters/sage_office/gray_floor.py --source assets/SAGE-10k/office-acb3ece0-background-v1 --output assets/SAGE-10k/office-acb3ece0-background-v2-gray
source adapters/sage_robot/env.sh
envs/sage-robot/bin/python adapters/sage_office/preview.py --background "$PWD/assets/SAGE-10k/office-acb3ece0-background-v2-gray" --output "$PWD/experiments/phase8-office05-gray-20261108-153821" --full-room --headless --enable_cameras --experience "$PWD/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit"
```

[Full room](../../experiments/phase8-office05-gray-20261108-153821/room_topdown.png), [oblique cutaway](../../experiments/phase8-office05-gray-20261108-153821/office_overview.png), [audit](../../experiments/phase8-office05-gray-20261108-153821/verification.json). New background SHA256: `eec0fcd5f6a84515fe81d6b3d15a9d69a63c27de481671db6ab2d41dfea22c1d`.

### 2026-11-08: Franka front-left tabletop camera

Added explicit `configs/tabletop-front-left-v1.json`: world position [0.75,2.60,2.50]m, target [2.08,1.92,1.04]m, downward pitch44.35 degrees. LIBERO-inspired composition adapted to the current table. Existing128x128 tabletop intrinsics retained; a separate768x768 preview camera shares its pose/FOV. Office runtime accepts `tabletop_profile`; preview CLI accepts `--tabletop-profile` and reapplies it after restoring historical camera states. Selected-office records the profile, which downstream workflows must explicitly apply.

Final run `phase8-office05-camera-20261108-155226` visually verified: robot/gripper/milk/task region visible. Earlier155128 framing retained for provenance. Native/background hashes unchanged,100-step hold trace identical to gray-floor run,4.01micrometer maximum milk displacement. Exit0; actual pose/direction, normalized HD/policy intrinsics, original policy intrinsics and projected keypoints audited in `verification.json`. Wrist configuration unchanged; GPU process exited. No training, demonstration collection or policy evaluation. New visual data must be collected/rerendered before training in this view; historical data/checkpoints remain frozen.

Reproduce after sourcing `adapters/sage_robot/env.sh` (use a fresh output directory):

```bash
envs/sage-robot/bin/python adapters/sage_office/preview.py --background "$PWD/assets/SAGE-10k/office-acb3ece0-background-v2-gray" --tabletop-profile "$PWD/configs/tabletop-front-left-v1.json" --output "$PWD/experiments/NEW-CAMERA-PREVIEW" --headless --enable_cameras --experience "$PWD/envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit"
```

[768x768 camera preview](../../experiments/phase8-office05-camera-20261108-155226/tabletop_preview.png), [actual128x128 policy input](../../experiments/phase8-office05-camera-20261108-155226/tabletop.png), [audit](../../experiments/phase8-office05-camera-20261108-155226/verification.json). Reference: [LIBERO camera configuration](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/bddl_base_domain.py). This is an adapted composition, not a claim of identical LIBERO camera calibration.

### 2026-11-08: Table-front center-axis camera from user reference

User clarified front/back and supplied `1.png`. Selected profile now `configs/tabletop-front-center-v2.json`: camera [2.08,2.95,2.30]m, look-at [2.08,1.90,1.30]m, sharing X=2.08 with table/Franka base. Downward pitch43.60 degrees; focal24mm (previous32), horizontal FOV47.17 degrees. Optional profile focal length is applied before cloning the HD camera. Policy128x128 and HD768x768 share pose/FOV; wrist unchanged.

Final run `phase8-office05-center-20261108-160626` passes render/exit checks, pose/center-axis projection checks, HD/policy calibration matching, and unchanged native/background hashes.100-step hold exactly matches previous run (milk4.01micrometer displacement). GPU process exited. First160416 preview retained: front divider occluded the milk. Current framing shows complete milk and centered robot, but divider still partially hides tissue/tray; the44-degree pitch is a compromise, not a shallow view. No object moves/hiding, collection/training/policy evaluation. New visuals require explicit use of the selected camera profile.

Use the previous preview command with `--tabletop-profile "$PWD/configs/tabletop-front-center-v2.json"` and a fresh output directory. [Current HD view](../../experiments/phase8-office05-center-20261108-160626/tabletop_preview.png), [128px policy view](../../experiments/phase8-office05-center-20261108-160626/tabletop.png), [verification](../../experiments/phase8-office05-center-20261108-160626/verification.json). Prior left-front profile is retained for reproducibility.

### 2026-11-08: Tabletop camera15% closer

Selected `configs/tabletop-front-center-close-v3.json`: position[2.08,2.7925,2.15]m, same look-at[2.08,1.90,1.30]m, orientation, center axis and24mm focal length. Run `phase8-office05-close-20261108-161145` renders complete robot/milk with closer framing. Exit0; actual15% optical-axis movement, unchanged intrinsics/wrist, background/native asset records and identical100-step physics trace verified. Small artifacts synced locally; no collection/training/policy evaluation; GPU process exited.

Use `--tabletop-profile configs/tabletop-front-center-close-v3.json` with the existing preview command. [Current HD tabletop](../../experiments/phase8-office05-close-20261108-161145/tabletop_preview.png).

### 2026-11-08: Closer tabletop view with additional downward tilt

Selected `configs/tabletop-front-center-down-v4.json`: move12.25cm toward Franka to[2.08,2.67,2.15]m; target[2.08,1.90,1.22]m; pitch50.38 degrees(+6.77). Run `phase8-office05-down-20261108-161923` passes render/exit, actual pose, HD/policy calibration, unchanged intrinsics/wrist, background/native records and identical100-step state checks. Preview inspected: table/gripper/milk/tissue visible, uppermost robot shoulder slightly clipped at top; divider retained. No collection/training/policy evaluation; GPU process exited.

Use `--tabletop-profile configs/tabletop-front-center-down-v4.json`. [Current HD tabletop](../../experiments/phase8-office05-down-20261108-161923/tabletop_preview.png).
