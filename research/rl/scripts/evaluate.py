"""Independent LIBERO evaluation using RLinf's native policy and simulator wrapper."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np


def adapt_actions(actions):
    """Map published 0=closed/1=open gripper to robosuite +1=closed/-1=open."""
    result = np.asarray(actions).copy()
    result[..., -1] = np.where(result[..., -1] >= .5, -1., 1.)
    return result


def prepare_scratch(root, run_id):
    if Path(run_id).name != run_id:
        raise ValueError('Expected one run basename')
    base = (Path(root) / 'scratch/libero').resolve()
    target = base / run_id
    if target.is_symlink() or target.resolve().parent != base:
        raise ValueError('Scratch target escaped its parent')
    marker = target / '.evoworld-owned'
    if target.exists():
        if not marker.is_file() or marker.read_text() != run_id:
            raise ValueError('Refusing to claim an existing unowned scratch directory')
    else:
        target.mkdir(parents=True)
        marker.write_text(run_id)
    return target


def cleanup_scratch(root, run_id, *, passed):
    if not passed or Path(run_id).name != run_id:
        raise ValueError('Cleanup requires passed verification and one run basename')
    base = (Path(root) / 'scratch/libero').resolve()
    target = base / run_id
    if target.is_symlink() or target.resolve().parent != base:
        raise ValueError('Cleanup target escaped scratch')
    if not target.exists():
        return 0
    marker = target / '.evoworld-owned'
    if not marker.is_file() or marker.read_text() != run_id:
        raise ValueError('Missing ownership marker')
    size = sum(p.stat().st_size for p in target.rglob('*') if p.is_file() and not p.is_symlink())
    shutil.rmtree(target)
    return size


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['env', 'smoke', 'full'], required=True)
    parser.add_argument('--num-envs', type=int, default=20)
    args = parser.parse_args()
    root = Path(os.environ['EVOWORLD_ROOT'])
    run_id = os.environ['EVOWORLD_RUN_ID']
    output_root = Path(os.environ['EVOWORLD_RL_OUTPUT_ROOT'])
    out = output_root / 'runs/libero/pi05-sft' / run_id
    formal = out / 'metrics/full-episodes.jsonl'
    if (out / 'metrics/summary.json').exists() or (formal.exists() and formal.stat().st_size):
        raise RuntimeError('Existing formal results are immutable; use a new run ID')
    scratch = prepare_scratch(output_root, run_id)
    for name in ['config', 'metrics', 'videos', 'provenance', 'logs']:
        (out / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('HF_HOME', str(root / 'cache/huggingface'))
    os.environ.setdefault('ROBOT_PLATFORM', 'LIBERO')
    os.environ.setdefault('LIBERO_TYPE', 'standard')
    import torch
    import imageio.v2 as imageio
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    from rlinf.models.embodiment.openpi import get_model

    torch.set_num_threads(4)
    repo = Path(os.environ['EVOWORLD_RLINF_SOURCE'])
    ckpt = Path(os.environ['EVOWORLD_LIBERO_CHECKPOINT'])
    os.environ['PI05_FEWSHOT_CHECKPOINT_DIR'] = str(ckpt)
    env_cfg = OmegaConf.load(repo / 'examples/embodiment/config/env/libero_10.yaml')
    env_cfg = OmegaConf.merge(env_cfg, {
        'seed': 0, 'total_num_envs': args.num_envs, 'is_eval': True,
        'auto_reset': True, 'ignore_terminations': True,
        'use_fixed_reset_state_ids': True, 'use_ordered_reset_state_ids': True,
        'max_episode_steps': 520, 'max_steps_per_rollout_epoch': 520 * math.ceil(500 / args.num_envs),
        'video_cfg': {'save_video': False, 'video_base_dir': str(scratch / 'video')},
        'share_memory': False, 'skip_intermediate_renders': True,
    })
    if args.stage == 'full' and (args.num_envs > 500 or 500 % args.num_envs):
        raise ValueError('Full evaluation requires a parallelism divisor of 500')
    env = LiberoEnv(env_cfg, args.num_envs, 0, 1, None)
    writers = {}
    start = time.monotonic()
    try:
        obs, _ = env.reset()
        assert obs['states'].shape == (args.num_envs, 8)
        assert obs['main_images'].shape == (args.num_envs, 256, 256, 3)
        assert obs['wrist_images'].shape == obs['main_images'].shape
        for i in range(min(args.num_envs, 10)):
            imageio.imwrite(out / 'videos' / f'initial-task-{env.task_ids[i]:02d}.png', np.concatenate([obs['main_images'][i].numpy(), obs['wrist_images'][i].numpy()], axis=1))
        if args.stage == 'env':
            before = obs['states'][:, -2:].numpy().copy()
            a = np.zeros((args.num_envs, 5, 7), dtype=np.float32)
            # Direct physics command +1 must close the gripper.
            a[..., -1] = 1
            for _ in range(4):
                obs_list, *_ = env.chunk_step(a)
            closed = obs_list[-1]['states'][:, -2:].numpy().copy()
            a[..., -1] = -1
            for _ in range(4):
                obs_list, *_ = env.chunk_step(a)
            opened = obs_list[-1]['states'][:, -2:].numpy().copy()
            assert np.abs(opened).sum(axis=-1).mean() > np.abs(closed).sum(axis=-1).mean() + .01
            write_json(out / 'metrics/env-check.json', {'passed': True, 'before': before.tolist(), 'closed': closed.tolist(), 'opened': opened.tolist()})
            freed = cleanup_scratch(output_root, run_id, passed=True)
            write_json(out / 'metrics/env-cleanup.json', {'passed': True, 'removed_bytes': freed, 'scope': str(scratch)})
            print('ENV CHECK PASSED', flush=True)
            return

        model_cfg = OmegaConf.load(repo / 'examples/embodiment/config/model/pi0_5.yaml')
        public = OmegaConf.load(ckpt / 'rlinf_model_config.yaml').rollout.model
        model_cfg = OmegaConf.merge(model_cfg, public, {'openpi': {'task': 'eval', 'torch_compile': False}})
        # The factory defaults to float32 parameters, as in the published fragment.
        OmegaConf.save(OmegaConf.create({'env': env_cfg, 'model': model_cfg}), out / 'config' / f'{args.stage}-resolved.yaml', resolve=True)
        model = get_model(model_cfg).eval().to('cuda')
        torch.manual_seed(195)
        torch.cuda.manual_seed_all(195)
        rng = torch.Generator(device='cuda').manual_seed(195)
        records = []
        max_chunks = 10 if args.stage == 'smoke' else 104 * (500 // args.num_envs)
        first_actions = None
        records_path = out / 'metrics' / f'{args.stage}-episodes.jsonl'
        if args.stage == 'full' and records_path.exists() and records_path.stat().st_size:
            raise RuntimeError('Refusing to overwrite existing formal episodes')
        with records_path.open('w') as episodes:
            for chunk in range(max_chunks):
                previous_tasks = list(env.task_ids)
                previous_trials = list(env.trial_ids)
                if chunk < 104:
                    for i in range(min(args.num_envs, 10)):
                        if i not in writers:
                            video_dir = out / 'videos' if args.stage == 'full' else scratch
                            writers[i] = imageio.get_writer(video_dir / f'{args.stage}-task-{previous_tasks[i]:02d}-trial-{previous_trials[i]:02d}.mp4', fps=4)
                        frame = np.concatenate([obs['main_images'][i].numpy(), obs['wrist_images'][i].numpy()], axis=1)
                        writers[i].append_data(frame)
                elif writers:
                    for w in writers.values():
                        w.close()
                    writers.clear()
                with torch.inference_mode():
                    actions, _ = model.predict_action_batch(obs, mode='eval', rng=rng)
                raw = actions.detach().float().cpu().numpy()
                assert raw.shape == (args.num_envs, 5, 7), raw.shape
                assert np.isfinite(raw).all()
                converted = adapt_actions(raw)
                if first_actions is None:
                    first_actions = {'raw': raw.tolist(), 'env': converted.tolist()}
                    write_json(out / 'metrics' / f'{args.stage}-first-actions.json', first_actions)
                obs_list, _, terms, truncs, infos = env.chunk_step(converted)
                obs = obs_list[-1]
                info = infos[-1]
                if 'final_info' in info:
                    mask = np.asarray(info['_final_info'])
                    epi = info['final_info']['episode']
                    for i in np.flatnonzero(mask):
                        rec = {'task_id': int(previous_tasks[i]), 'trial_id': int(previous_trials[i]), 'success': bool(epi['success_once'][i]), 'steps': int(epi['episode_len'][i])}
                        records.append(rec)
                        episodes.write(json.dumps(rec) + '\n')
                    episodes.flush()
                if (chunk + 1) % 104 == 0 or chunk == max_chunks - 1:
                    progress = {'stage': args.stage, 'chunks': chunk + 1, 'episodes': len(records), 'successes': sum(x['success'] for x in records), 'elapsed_seconds': time.monotonic()-start, 'max_cuda_allocated_gb': torch.cuda.max_memory_allocated()/1e9}
                    write_json(out / 'metrics' / f'{args.stage}-progress.json', progress)
                    print(json.dumps(progress), flush=True)
        if args.stage == 'full':
            pairs = {(r['task_id'], r['trial_id']) for r in records}
            assert len(records) == 500 and pairs == {(t, s) for t in range(10) for s in range(50)}
            tasks = [{'task_id': t, 'name': env.task_suite.get_task(t).language, 'episodes': 50, 'successes': sum(r['success'] for r in records if r['task_id'] == t)} for t in range(10)]
            successes = sum(r['success'] for r in records)
            summary = {'status': 'complete', 'episodes': 500, 'successes': successes, 'success_rate': successes/500, 'published_success_rate': .566, 'tasks': tasks, 'elapsed_seconds': time.monotonic()-start, 'max_cuda_allocated_gb': torch.cuda.max_memory_allocated()/1e9, 'protocol_notes': ['single GPU; parallel batch size differs from publisher', 'RLinf native Pi0Eval and LiberoEnv; explicit gripper conversion', 'Euler eval; Flow-SDE noise level is a training parameter, not extra eval noise']}
            write_json(out / 'metrics/summary.json', summary)
            (out / 'README.md').write_text('# LIBERO-10 独立基线评测\n\n' + f'π0.5 few-shot 20k：{successes}/500 = {successes/5:.1f}%。\n\n' + '正式指标见 metrics/summary.json；逐回合记录见 metrics/full-episodes.jsonl；配置见 config/；前十个回合双视角视频见 videos/。\n')
            print('FORMAL EVALUATION VERIFIED', flush=True)
        else:
            write_json(out / 'metrics/policy-check.json', {'passed': True, 'chunks': max_chunks, 'actions_finite': True})
            print('POLICY CHECK PASSED', flush=True)
    finally:
        for w in writers.values():
            w.close()
        env.env.close()
    freed = cleanup_scratch(output_root, run_id, passed=True)
    write_json(out / 'metrics' / f'{args.stage}-cleanup.json', {'passed': True, 'removed_bytes': freed, 'scope': str(scratch)})


if __name__ == '__main__':
    main()
