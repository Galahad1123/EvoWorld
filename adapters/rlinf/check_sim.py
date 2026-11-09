"""Smoke-test RLinf's official task wrapper; no policy or pretrained weights."""
import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def exercise(env, out):
    obs, _ = env.reset(seed=42)
    for _ in range(5):
        actions = torch.zeros((1, 7), device=env.device)
        actions[:, -1] = 1
        obs, reward, done, truncated, info = env.step(actions)
        assert torch.isfinite(reward).all()
        assert torch.isfinite(obs['states']).all()
    image_stats = {}
    for key in ['main_images', 'wrist_images']:
        pixels = obs[key][0].detach().cpu().numpy()
        assert pixels.shape == (256, 256, 3), (key, pixels.shape)
        assert pixels.std() > 5, (key, pixels.std())
        image_stats[key] = {'std': float(pixels.std()), 'shape': list(pixels.shape)}
        Image.fromarray(pixels.astype('uint8')).save(out / (key + '.png'))
    reset_obs, _ = env.reset(seed=43)
    assert torch.isfinite(reset_obs['states']).all()
    return {
        'passed': True, 'num_envs': 1, 'steps': 5,
        'observation_shapes': {k: list(v.shape) for k, v in obs.items() if isinstance(v, torch.Tensor)},
        'image_stats': image_stats, 'reset_tested': True, 'model_policy_used': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/rlinf-setup' / ('sim-smoke-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from rlinf.envs.sim.isaaclab.tasks.stack_cube import IsaaclabStackCubeEnv
    cfg = OmegaConf.load(ROOT / 'upstream/RLinf/examples/embodiment/config/env/isaaclab_stack_cube.yaml')
    cfg.total_num_envs = cfg.init_params.num_envs = 1
    cfg.max_episode_steps = 20
    cfg.video_cfg.video_base_dir = str(out)
    cfg.video_cfg.save_video = False
    env = None
    try:
        env = IsaaclabStackCubeEnv(cfg, 1, 0, 1, None)
        result = exercise(env, out)
        result['task'] = cfg.init_params.id
    finally:
        gc.collect()
        if env is not None:
            torch.cuda.synchronize()
            env.close()
    result['close_completed'] = True
    (out / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    print('Results:', out, flush=True)
