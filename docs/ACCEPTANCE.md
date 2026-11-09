# Migration acceptance and limits

Checks performed without GPT requests, policy evaluation, SFT updates or RL iterations:

- Ten lightweight tests: config relocation, relative overrides, credential precedence without shell evaluation, preservation of venv interpreter symlinks, explicit Isaac experience selection, empty-config rejection, camera contract mismatch, corrupt resource rejection, dispatch construction and missing baseline rejection.
- Python compilation of the new runtime and current milk-bin / realtosim_photo adapters.
- Real pinned RLinf Hydra config composition: global/micro batch 4, 20,000 steps, explicit learning-rate floor ratio 0.1.
- Actual converted base model SHA256 verified: `f6391204c480d6c54d31d63601e7963ef23301157df052a4b8183e7b019ce2f1`.
- D46 official OpenPI data loader read a real batch of four demonstrations; no gradients or optimizer steps.
- Expert source patch reverse-apply check passed against existing adapted source.
- Current private scene package: 190 files; ten USD layers rebased from absolute texture paths. USD inspection found zero missing dependencies and zero remaining absolute asset references. Updated assets/manifests copied to both hosts.
- Read-only capability probes distinguish prerequisite availability from runtime verification. The source host has the expert/SFT/conversion environments; its GPT authoring uses the SAGE Python, not the minimal USD environment. D46 lacks the GPT/expert environments and M2T2 weights, while SFT/conversion prerequisites exist. No formal checkpoint is selected automatically.

Fresh code review identified explicit-machine-config, relative-argument, empty-doctor and RL-budget regressions; fixes are in this branch. Formal RL uses the original baseline competence gate, checkpoint hash matching, single-run lock, four-hour deadline and exact-marker subprocess cleanup. Existing 15% baseline does not meet the gate; migration does not authorize bypassing it.

Limits: no new simulation rendering, expert collection, GPT generation, SFT training or policy success evaluation was performed for this migration. The side70 camera is preview-only and must not be paired silently with existing center-camera data. Environment reconstruction instructions preserve pinned sources and patches, but installing fresh GPU extensions is hardware-dependent and has not been rerun from scratch on D46. Custom photo-derived scenes/datasets do not have a public download URL; their absence is explicit, not an automatic fallback to another scene.
