"""Sequential RECAP baseline with immutable recipes and a persistent time budget."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from recap_budget import (completed_checkpoint, finalize_policy_config, load_budget,
                          install_termination_handler, terminate_process_group)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--value-steps', type=int, default=8000)
    p.add_argument('--cfg-steps', type=int, default=10000)
    p.add_argument('--budget-hours', type=float, default=15)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--download-pid', type=int)
    a = p.parse_args()
    install_termination_handler()
    root = a.root.resolve()
    scripts = Path(__file__).resolve().parent
    py = root/'storage/envs/rlinf-libero-py312/bin/python'
    out = root/'research/rl/outputs/runs/libero/recap/pi05-20k-task0-20261110'
    out.mkdir(parents=True, exist_ok=True)
    status = out/'pipeline-status.json'
    lock = (out/'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status.exists() and not a.resume:
        raise FileExistsError('Existing pipeline status; use --resume')
    if (out/'metrics/summary.json').exists():
        print('Pipeline already complete')
        return
    if (out/'provenance/recipe.json').exists():
        recipe = json.loads((out/'provenance/recipe.json').read_text())
        if recipe['value_steps'] != a.value_steps or recipe['cfg_steps'] != a.cfg_steps:
            raise ValueError('Resume budget differs from immutable recipe')
        if recipe.get('policy_batch') != 16 or recipe.get('policy_micro_batch') != 4:
            raise ValueError('Existing recipe predates approved global16/micro4 budget')
    budget = None

    def record(stage, **kwargs):
        status.write_text(json.dumps({'stage': stage, 'time': time.time(), **kwargs}, indent=2))
        print(stage, kwargs, flush=True)

    def execute(cmd, stage, env=None):
        if budget and time.time() >= budget['deadline_unix']:
            raise TimeoutError('RECAP total time budget expired before '+stage)
        record(stage, command=list(map(str, cmd)))
        (out/'logs').mkdir(exist_ok=True)
        with (out/'logs'/f'controller-{stage}.log').open('a') as log:
            proc = subprocess.Popen(list(map(str, cmd)), env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while proc.poll() is None:
                    if budget and time.time() >= budget['deadline_unix']:
                        raise TimeoutError('RECAP total time budget expired during '+stage)
                    with (out/'resources.jsonl').open('a') as f:
                        gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                        f.write(json.dumps({'time': time.time(), 'stage': stage, 'gpu': gpu.stdout.strip(),
                                            'disk_free': shutil.disk_usage(root).free})+'\n')
                    time.sleep(10)
            except BaseException:
                terminate_process_group(proc)
                raise
            if proc.returncode:
                raise RuntimeError(f'{stage} exited {proc.returncode}; see {log.name}')

    def stage(name, run=out):
        execute([py, scripts/'recap_stage.py', '--root', root, '--run', run, '--stage', name,
                 *(['--resume'] if a.resume else [])], name if run == out else 'calibration')

    try:
        record('waiting_for_verified_resources')
        receipts = [root/'storage/datasets/recap-libero-task0/verified.json',
                    root/'storage/models/gemma-3-270m/verified.json',
                    root/'storage/models/siglip2-so400m-patch14-224/verified.json']
        download_deadline = time.time()+12*3600
        while not all(x.is_file() for x in receipts):
            if a.download_pid:
                try:
                    os.kill(a.download_pid, 0)
                except ProcessLookupError:
                    raise RuntimeError('Downloader exited without verified resources')
            if time.time() > download_deadline:
                raise TimeoutError('Resource download not verified within 12h')
            time.sleep(30)
        smoke = root/'research/rl/outputs/scratch/recap-20261110/smoke'
        assert json.loads((smoke/'metrics/acceptance.json').read_text())['passed'], 'Smoke acceptance required'
        budget = load_budget(out/'provenance/time-budget.json', a.budget_hours)
        if not (out/'provenance/recipe.json').exists():
            execute([py, scripts/'recap_prepare.py', '--root', root, '--mode', 'full',
                     '--value-steps', a.value_steps, '--cfg-steps', a.cfg_steps,
                     '--deadline-unix', budget['training_deadline_unix']], 'prepare')
        stage('returns')
        execute([py, scripts/'recap_video_cache.py', '--source', root/'storage/datasets/recap-libero-task0',
                 '--dest', root/'storage/cache/recap-libero-task0-gop8', '--workers', 16,
                 '--activate-run', out], 'video-cache')
        stage('value')
        value_checkpoint = completed_checkpoint(out/'value')
        # The value deadline can shorten training; advantages must use the actual saved model.
        import yaml
        adv_path = out/'config/advantages.yaml'
        adv = yaml.safe_load(adv_path.read_text())
        if adv['advantage']['value_checkpoint'] != str(value_checkpoint):
            if (out/'provenance/advantages-execution.json').exists():
                raise ValueError('Cannot change value checkpoint after advantage computation started')
            shutil.copy2(adv_path, out/'config/advantages-requested.yaml')
            adv['advantage']['value_checkpoint'] = str(value_checkpoint)
            adv_path.write_text(yaml.safe_dump(adv, sort_keys=False))
        stage('advantages')

        calibration = root/'research/rl/outputs/scratch/recap-20261110/calibration-global16'
        if not (out/'provenance/policy-budget.json').exists():
            (calibration/'config').mkdir(parents=True, exist_ok=True)
            cpath = calibration/'config/cfg.yaml'
            if not cpath.exists():
                cfg = yaml.safe_load((out/'config/cfg.yaml').read_text())
                cfg['runner'].update(max_steps=100, save_interval=-1, val_check_interval=-1)
                cfg['runner']['logger'].update(log_path=str(calibration), experiment_name='cfg')
                cfg['actor']['optim'].update(total_training_steps=100, lr_warmup_steps=10)
                cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
            stage('cfg', calibration)
            report = json.loads((calibration/'cfg/training_status.json').read_text())
            if report['global_step'] != 100 or report['stop_reason'] != 'max_steps':
                raise TimeoutError('Calibration did not finish within the training budget')
            shutil.copy2(calibration/'cfg/training_status.json', out/'provenance/calibration.json')
        report = json.loads((out/'provenance/calibration.json').read_text())
        effective = finalize_policy_config(out, report, budget['deadline_unix'], a.cfg_steps)
        print('Calibrated policy budget:', effective, flush=True)
        stage('cfg')
        trained = completed_checkpoint(out/'cfg')/'model_state_dict'
        package = out/'policy'
        package.mkdir(exist_ok=True)
        link = package/'model_state_dict'
        if link.exists() and link.resolve() != trained.resolve():
            raise ValueError('Packaged policy differs from completed checkpoint')
        if not link.exists():
            link.symlink_to(trained, target_is_directory=True)
        base = root/'storage/models/pi05-fewshot-libero/global_step_20000'
        shutil.copytree(base/'physical-intelligence', package/'physical-intelligence', dirs_exist_ok=True)
        env = dict(os.environ)
        env.update(PYTHONPATH=str(root/'third_party/RLinf'), HF_HOME=str(root/'storage/cache/huggingface'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                   LIBERO_TYPE='standard', ROBOT_PLATFORM='LIBERO', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
        for label, checkpoint, guidance in [('baseline', base, 'no_guide'), ('final', package, 'positive')]:
            evaluation = out/f'evaluation-{label}'
            if (evaluation/'metrics/summary.json').is_file():
                continue
            if evaluation.exists():
                evaluation.rename(out/'provenance'/f'failed-evaluation-{label}-{time.time_ns()}')
            execute([py, scripts/'recap_eval.py', '--root', root, '--checkpoint', checkpoint,
                     '--output', evaluation, '--guidance', guidance], f'evaluation-{label}', env)
        b = json.loads((out/'evaluation-baseline/metrics/summary.json').read_text())
        f = json.loads((out/'evaluation-final/metrics/summary.json').read_text())
        vr = json.loads((out/'value/training_status.json').read_text())
        pr = json.loads((out/'cfg/training_status.json').read_text())
        summary = {'complete': True, 'baseline': b, 'recap': f,
                   'delta_percentage_points': 100*(f['success_rate']-b['success_rate']),
                   'requested_value_steps': a.value_steps, 'requested_cfg_steps': a.cfg_steps,
                   'value_steps': vr['global_step'], 'cfg_steps': pr['global_step'],
                   'budget_limited': vr['global_step'] < a.value_steps or pr['global_step'] < a.cfg_steps,
                   'time_budget': budget, 'finished': time.time(), 'policy_budget': effective,
                   'notes': ['Public SFT covers 10 tasks; rollout/eval data are Task0 only.',
                             '20k initialization; not a numerical reproduction of the published 48.8% baseline.',
                             '50 fixed Task0 reset states, 520 steps, action chunk 5.']}
        (out/'metrics/summary.json').write_text(json.dumps(summary, indent=2))
        (out/'README.md').write_text(f"# RECAP Task 0 基线\n\n初始化：{b['successes']}/50；RECAP：{f['successes']}/50。\n\n"
                                    f"实际价值步数：{vr['global_step']}；实际策略步数：{pr['global_step']}；"
                                    f"预算截断：{summary['budget_limited']}。\n\n配置见 config，指标见 metrics/summary.json，视频见 evaluation-*/videos。\n")
        record('complete', summary='metrics/summary.json')
    except Exception as exc:
        record('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
