[中文（优先维护）](ENVIRONMENTS.md)

# Environment and source reconstruction

Start with `python scripts/prepare_sources.py --destination /your/storage/upstream` to inspect exact source URLs/commits/patches. Add `--execute` to clone and apply them. Existing checkouts at other commits are never reset. Patches are idempotently checked; any conflicting local modifications stop preparation. `sage-expert` is a separate patched source copy so original SAGE stays available.

Keep separate environments. The lightweight launcher needs only Python >=3.10 and PyYAML. Existing validated component interpreters can be reused by setting their exact executable paths (do not resolve symlinks outside a virtualenv).

- **GPT authoring**: OpenAI SDK and PyYAML; existing deployment uses the SAGE environment for the API process. The minimal g1-realtosim environment instead provides USD/geometry tools and is not automatically a GPT SDK environment. Generation and execution must use separate credential scopes. `realtosim_photo/build_candidate.py` strips secret variables before executing generated code.
- **Expert generation**: original SAGE robot environment with M2T2/PointNet++, cuRobo, the patched SAGE IsaacLab namespace, and compatible Torch/CUDA. Follow pinned SAGE/M2T2/cuRobo installation instructions and retain their licenses. The environment's extensions must be rebuilt on the destination hardware; copying a venv is unsupported. Link the pinned `m2t2.pth` to the location expected by the M2T2 source. Place the selected SAGE fixture under the server's results directory as required by its original collector.
- **RLinf/SFT/convert**: use the pinned RLinf embodied installation instructions and the captured installer patch; installed versions from the original host are recorded under `adapters/rlinf/requirements-frozen.txt`. That file is evidence, NOT a clean portable pip lock; editable entries must be installed from the selected upstream paths. Current tested D46 environment is Python 3.12 / Torch 2.11 CUDA 13.0. No standalone SFT smoke training is required for migration.
- **Isaac runtime**: may be a standalone interpreter or shared libraries used by another interpreter. Set `environments.sim` and optionally `isaac_experience`. Discovery runs the selected interpreter to locate the experience, never hardcodes Python's minor version.
- **RoboTwin**: optional and separate from Isaac; use the official RLinf-compatible branch and matching assets/model protocol. Installation and evaluation are not part of this migration.

`doctor` probes package availability without importing GPU libraries. `prerequisites_available` is not a claim of simulation or training acceptance. Keep capability reports explicit: an installation can support SFT while lacking M2T2 or the GPT authoring SDK.

When changing CUDA/Torch/Isaac versions, preserve the previous environment and rerun relevant compatibility checks rather than upgrading it in place. NVIDIA proprietary packages/assets require their own license acceptance; the launcher does not silently accept licenses.
