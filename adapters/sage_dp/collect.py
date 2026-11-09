"""Execute a recorded, minimally instrumented copy of the original SAGE collector."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'adapters/sage_robot/src/server/isaaclab/data_generation_franka_pnp_from_layout_parsing.py'
out = Path.cwd()
seed = int(os.environ.get('EVOWORLD_DP_SEED', '81'))
text = source.read_text()
replacements = [
    ('import random\n', f'import random\nrandom.seed({seed}); np.random.seed({seed}); torch.manual_seed({seed})\nimport runtime as dp_runtime\n'),
    ('    env = gym.make(args_cli.task, cfg=env_cfg).env',
     f'    dp_runtime.configure(env_cfg, {seed})\n    env = gym.make(args_cli.task, cfg=env_cfg).env\n    dp_runtime.attach(env)'),
    ('            obs_dict = env.observation_manager.compute()',
     '            if iteration == T_init + 1:\n                dp_runtime.snapshot(env, current_data_idx, target_object_name, place_location_w, ".")\n            obs_dict = env.observation_manager.compute()'),
    ('            dones = terminated | truncated',
     '            dones = terminated | truncated\n            dp_runtime.trace(env, current_data_idx, iteration, target_object_name, actions, ".")'),
    ('                        collector_interface.flush()',
     '                        collector_interface.flush()\n                        with open("demo-attempts.jsonl", "a") as dp_map:\n                            dp_map.write(json.dumps({"demo": collected_data_episodes, "attempt": current_data_idx - 1}) + "\\n")'),
]
for old, new in replacements:
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new)
(out / 'collector-executed.py').write_text(text)
(out / 'collector-source.json').write_text(json.dumps({
    'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'executed_sha256': hashlib.sha256(text.encode()).hexdigest(), 'seed': seed,
    'scope': 'seed, existing wrist fix, disable debug frames, read-only evidence; original planner and collector unchanged'
}, indent=2))
exec(compile(text, str(source), 'exec'), {'__name__': '__main__', '__file__': str(source)})
