# Resources and downloads

Git contains code, not weights, datasets, photos, generated scenes or environments. Downloads are explicit; `doctor` never downloads. Configure actual paths in the ignored machine YAML `resources` mapping. Keep `.partial` downloads unavailable until checksums pass.

| Resource ID | Official source / download | Version and preparation |
|---|---|---|
| `base_model` | [OpenPI π0.5 base](https://storage.googleapis.com/openpi-assets/checkpoints/pi05_base/) (`gs://openpi-assets/checkpoints/pi05_base`) / [OpenPI instructions](https://github.com/Physical-Intelligence/openpi) | Download the complete JAX checkpoint; convert with the pinned RLinf `rlinf/utils/ckpt_convertor/openpi/jax_to_openpi.py`. The GCS prefix is not an HTML file index. Use the official downloader or GCS tool. This is NOT a ready-made PyTorch weight URL. Conversion used 667 tensors / 3,353,433,872 parameters. Task norm_stats must come from training demonstrations. |
| `m2t2` | [Official model page](https://huggingface.co/wentao-yuan/m2t2), [pinned weight file](https://huggingface.co/wentao-yuan/m2t2/resolve/a20325dcd19cd1b838274974d0ebe35bfe383796/m2t2.pth) | SHA256 `e35c3cb11e06f46c5d406bdfc756bc06f48b256dd6d638408d9a5ff13deb97fb`. Mirror: substitute `https://hf-mirror.com` for the HF hostname; retain revision and verify hash. |
| SAGE-10k | [Official dataset](https://huggingface.co/datasets/nvidia/SAGE-10k/tree/d425de49ff445fd3903d790a11461d23ffd0c7dd) | Pinned dataset revision shown in URL. Current background derives from `layout_acb3ece0`. Download selected dependencies, not all 10k scenes. Follow upstream asset licensing. |
| `isaac_assets` | [NVIDIA Isaac 4.2 asset root](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/) | Fetch starting with `Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd` and referenced files. `adapters/sage_robot/fetch_assets.py` records file URLs and hashes; legacy launcher needs migration before use. |
| `photo_scene` | Generated locally; no public download | Scene bundle containing `scene_spec.json`, referenced USD/mesh/texture assets. GPT input photos are private. Preserve the existing phase-5/7 accepted bundle or regenerate via realtosim_photo generation → build → render → critique → physical verification. |
| `storage_box` | Generated locally; no public download | USD box generated from user photos with known dimensions. Point to `storage_box.usda`; retain its referenced assets and provenance privately. |
| `office_background` | Derived locally from pinned SAGE-10k above | Selected office with the approved gray floor. Original dataset download alone does not reproduce the local material modification. Preserve the modified bundle. |
| `env_template` | Generated locally; no public download | Accepted SAGE `env_init.yaml`; required by the custom simulation task. Must be packaged with its referenced scene dependencies. |
| `expert_source` | [Pinned SAGE](https://github.com/NVlabs/sage/tree/fe3966013a495dc1642710de93a83ddf66e534a0), [cuRobo](https://github.com/NVlabs/curobo/tree/d64c4b005459db10c5dd867d8b30a87d5bda9bdb) | Copy SAGE IsaacLab/M2T2/server plus cuRobo into an isolated source tree; apply `patches/expert-current.patch` at its root. Install PointNet++/cuRobo against the target environment; do not copy binary extensions across machines. |
| `train_data`, validation data | Generated locally; no public download | Collect expert HDF5, verify successes and dwell, then export with `export_dataset.py`. Keep split definitions and training-only normalization. |
| `checkpoint` | Trained locally; no public download | Explicit SFT checkpoint compatible with robot/action/camera contract; never silently select latest. |

The π0.5 conversion module used in the original run had SHA256 `e95be73afc0acf1b82a8ad0ba52a1967e083947829a2a10a50ba95c21db558d2`. Base-weight hashes must be recorded after conversion; JAX and PyTorch files do not share hashes.

Additional optional environments: [RoboTwin RLinf_support](https://github.com/RoboTwin-Platform/RoboTwin/tree/RLinf_support) and its official asset instructions; [RLinf RoboTwin evaluation](https://rlinf.readthedocs.io/en/latest/rst_source/evaluations/guides/robotwin.html). These are not installed by core setup. A single-arm milk-bin policy is not a drop-in dual-arm benchmark checkpoint.

Qwen weights are not required for the GPT-driven generation route. No private API key, VPN credential, asset access token, or user photograph is distributed.

## Current private scene package

The deployment migration packages the accepted resources as `assets/evoworld/milk-bin-v1/` with `photo_scene/`, `box/`, `office/`, `fixture/`, relative `env_init.yaml`, and a 190-file SHA256 manifest. This package remains outside Git on each host. Its `env_init.yaml` uses paths relative to the package; the simulator adapter resolves them at launch. Current deployments therefore no longer obtain these scene assets from phase output directories. This is private project data, not a public downloadable benchmark.

Example explicit source downloads (run only when resources are needed):

```bash
# GCS checkpoint directory; requires Google's gsutil, not an HTTP directory fetch.
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_base /your/model/storage/
# M2T2: pinned HF revision; HF_ENDPOINT can select hf-mirror.com.
hf download wentao-yuan/m2t2 m2t2.pth --revision a20325dcd19cd1b838274974d0ebe35bfe383796 --local-dir /your/model/storage/m2t2
```

Do not download Qwen for the GPT authoring workflow. Mirror failures do not change the expected revision or checksum. Gated assets require authorization through their original provider; credentials stay on the machine.
