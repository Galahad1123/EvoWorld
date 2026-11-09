# EvoWorld minimal scene-generation chain

This adapter runs **gpt-6-astra Responses → real MCP stdio tools → new SAGE FloorPlan → official SAGE-10k mesh/Isaac utilities → physics and USD reload → rendered-image review by GPT**.

Scope: one bounded, closed test room with one table, two chairs and one tabletop mug. Meshes, PBR parameters and materials are reused unchanged from the downloaded official SAGE-10k fixture. Room dimensions, object choices, positions and headings are generated anew. This is not the original heavy `layout_wo_robot.py` service or a TRELLIS/MatFuse reproduction; robot control is not included. The optional custom prompt must stay within this four-object contract.

## Run on 5090-1

```bash
cd ~/lizk/vlas/EvoWorld
envs/sage/bin/python scripts/with_gpt_env.py -- \
  envs/sage/bin/python -u adapters/minimal_sage/generate.py
```

The existing project launcher loads the remote `.env` and Clash proxy. The API key is not passed to the MCP/simulation child processes. Defaults: `gpt-6-astra`, low reasoning, at most 10 Responses calls and 3 simulator attempts. Each invocation creates a fresh `experiments/sage-generated-.../` directory. No SAGE simulator needs to be started separately: the MCP tool starts its isolated headless Isaac process and then a second process to reload/render.

Run adapter tests without GPU or API requests:

```bash
PYTHONPATH=adapters/minimal_sage envs/sage/bin/python -m pytest adapters/minimal_sage/tests -q
```

The integration test starts the real MCP server and exports a temporary scene with the official geometry functions. It also proves that an unknown scene ID and premature completion are rejected.

## Acceptance and evidence

`EVOWORLD_MINIMAL_GENERATION_PASS` requires a persisted completion report, passing physics/reload reports, actual model tool calls and an image-review round. Process exit 0 alone is insufficient; SimulationApp shutdown can hide exceptions. Each simulation attempt is immutable. Failure logs and API responses are retained.

Physics checks: exactly four dynamic bodies; 1 meter per stage unit and Z up; 120 finite pose samples; upstream stability; ≤5 cm center displacement and ≤8° rotation change; floor/table support within 3 cm; observed motion; last ten samples bounded by 1 mm translation and 0.1° rotation per sample. These are short-run smoke-test limits, not long-horizon or robot-task quality certification. The render is the saved **initial layout**, explicitly labeled to GPT. Final support comes from measured transforms, not from the screenshot.

Final reviewed run: `experiments/sage-generated-20261107-192851-ei5oyvf5/`, scene `4aaa64082306`.

- 19 tests passed; four actual GPT/MCP rounds completed.
- Four dynamic bodies stable; maximum center displacement 0.00393335 m, maximum rotation change 0.449369°.
- Mug/table bounding gap approximately -0.000454 m; support checks pass within the documented tolerance.
- Motion observed; last-ten-sample motion bounds pass; independent USD reload/render passes.
- Usage for this final run: 9410 input tokens and 397 output tokens, from API usage records. Earlier probes/attempts are separate.

Run-level files: `manifest.json`, `prompt.txt`, `tools.json`, `request-*.json`, `response-*.json`, `tool-*.json`, `completion.json`, `summary.json`, `verification-summary.json`. No API headers or credentials are recorded.

Scene-level files: `layout.json`, `plan.json`, `provenance.json`, `objects/`, `materials/`, `layout.glb`, `scene.usda`, `physics.json`, `render.json`, `preview.png`, and logs. The archived `generated-scene.zip` contains the self-contained scene directory, including relative texture references and evidence; ZIP integrity was verified. `scene.usda` stores the initial layout; trajectories are in `physics.json`.

The original source checkout is clean at `fe3966013a495dc1642710de93a83ddf66e534a0`; dataset kit revision is `d425de49ff445fd3903d790a11461d23ffd0c7dd`. Manifest and provenance files record script/input hashes. The full original SAGE server and from-scratch asset-generation backends remain outside this acceptance claim.

API references: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling) and [gpt-6-astra](https://developers.openai.com/api/docs/models/gpt-6-astra). Actual account/tool access was verified by the recorded API runs.

## Bounded Phase 2 protocol

`envs/sage/bin/python scripts/with_gpt_env.py -- envs/sage/bin/python -u scripts/run_phase2_small.py` creates exactly two sequential new layouts, each capped at6GPT requests,1simulation attempt,600explicit physics steps and360wall seconds. Do not run again merely to replace failures. `scripts/review_phase2_results.py BATCH_DIR` performs CPU-only reanalysis.

New simulation uses `fixed_step.py`, with explicit60Hz steps and measured callback/clock counters; initialization steps/time are recorded separately.600steps yield601samples and10seconds. The current tail check uses transformed mesh reference centers and61samples for the last full second; physical origin translation is not treated as body-center translation. Historical original minimal-run reports retain their old tracker semantics.

The completed formal batch is `phase2-small-20261107-200457-n70snxiw`:2/2 bounded checks pass,8GPT calls,69.106s formal wall time, no replacements. Original reports/code remain under `executed-source/`; reviewed reports are separate. Final reviewed test suite:30passing tests. Image-conditioned scenes, robot tasks and long-horizon validity remain outside this test.
