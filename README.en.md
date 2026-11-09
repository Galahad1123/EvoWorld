[中文（优先维护）](README.md)

# EvoWorld

Research integration of GPT-assisted real-to-sim scene/asset generation, SAGE expert demonstrations, Isaac simulation, and RLinf SFT/RL.

## Portable setup

1. Read [resource sources and downloads](docs/RESOURCES.md). Install the pinned component environments separately; do not install all their dependencies into this lightweight launcher environment.
2. Install this launcher with `pip install -e .` (Python >=3.10).
3. Copy `configs/local.example.yaml` to `configs/local.yaml`. Set interpreter paths, upstream checkout paths and resource mappings. Relative paths resolve against the YAML file's directory, not the shell working directory.
4. Put `OPENAI_API_KEY=...` in `~/.config/evoworld/credentials.env` (mode 600), or supply it in the GPT process environment. Existing project `.env` may be referenced by `credentials_file`. Never commit credentials.
5. Run `python -m evoworld doctor`. The report checks configured paths; missing entries and component execution readiness require the deployment audit described below.
6. Inspect a command before execution, e.g. `python -m evoworld sft --dry-run`. `--config /path/to/machine.yaml` selects another machine profile. Arguments after `--` are forwarded to the underlying tool.

Public experiment parameters live in task configs; machine config only locates dependencies and resources. GPT model and reasoning settings live in `configs/gpt.yaml`. Network configuration has separate GPT proxy and download mirror fields. Credentials are loaded only by `generate`.

Commands: `generate`, `collect`, `convert`, `sft`, `rl`, `eval`. These dispatch the existing implementations, not replacements for their algorithms. Each real invocation creates a separate run record; dry-run starts no training, simulation, download or API call. The realtosim generate command requires its original run directory, round name and prompt arguments. Input photos remain external.

## Migration status

The portable entrypoints are implemented and checked without starting training or policy evaluation. Historical adapter scripts and README files retain experimental context; fixed-host shell automation is archived as text under `docs/legacy-launchers`. Use the new launcher for current deployment. Environments, private assets and task checkpoints are external prerequisites: cloning this repository alone does not provide them. See [environment preparation](docs/ENVIRONMENTS.md) and [migration acceptance](docs/ACCEPTANCE.md). Existing experimental outputs are preserved externally.

The side70 wrist profile is a diagnostic candidate, not the observation contract used to train existing checkpoints. It is disabled by default until trajectory-level camera acceptance and regenerated training observations are ready.

Upstream versions and patches: `manifests/upstreams.json`. Third-party repositories retain their original licenses; no blanket license grant is made for their contents. Private resources and machine profiles are excluded from the repository.
