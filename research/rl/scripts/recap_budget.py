"""Persistent wall-clock budgets for the RECAP experiment controller."""
import json
import math
import time
from pathlib import Path


def load_budget(path: Path, hours: float, now=None):
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError('Budget hours must be finite and positive')
    if path.exists():
        value = json.loads(path.read_text())
        if value['hours'] != hours:
            raise ValueError('Resume budget differs from recorded hours')
        return value
    now = time.time() if now is None else now
    value = {'hours': hours, 'started': now, 'deadline_unix': now+hours*3600,
             'training_deadline_unix': now+hours*3600-1800}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return value


def choose_policy_budget(target_steps, seconds_per_step, deadline, now=None):
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise ValueError('Calibration throughput must be finite and positive')
    now = time.time() if now is None else now
    # Reserve 30min for final save/evaluation, 10min for startup and checkpoint I/O.
    available = deadline - now - 2400
    steps = min(target_steps, int(available / (seconds_per_step * 1.15)) // 100 * 100)
    if steps < 100:
        raise TimeoutError('Insufficient remaining budget for policy training and evaluation')
    return {'steps': steps, 'warmup_steps': min(1500, int(steps*.15)),
            'seconds_per_step': seconds_per_step, 'estimated_at': now,
            'training_deadline_unix': deadline-1800, 'target_steps': target_steps}


def completed_checkpoint(stage: Path):
    report = json.loads((stage/'training_status.json').read_text())
    step = report['global_step']
    ckpt = stage/'checkpoints'/f'global_step_{step}'/'actor'
    if step <= 0 or not checkpoint_is_complete(ckpt):
        raise RuntimeError(f'Actual completed checkpoint is missing or incomplete: {ckpt}')
    return ckpt


def finalize_policy_config(out: Path, calibration: dict, deadline: float, target: int, now=None):
    """Freeze the effective schedule once, before any formal policy update."""
    import hashlib
    import statistics
    import yaml
    receipt = out/'provenance/policy-budget.json'
    config = out/'config/cfg.yaml'
    if receipt.exists():
        decision = json.loads(receipt.read_text())
        if hashlib.sha256(config.read_bytes()).hexdigest() != decision['config_sha256']:
            raise ValueError('Final policy configuration changed after calibration')
        return decision
    times = calibration['step_seconds'][20:]
    if len(times) < 50 or not all(math.isfinite(x) and x > 0 for x in times):
        raise ValueError('Calibration requires at least 50 finite post-warmup timings')
    decision = choose_policy_budget(target, statistics.mean(times), deadline, now)
    original = out/'config/cfg-requested.yaml'
    if not original.exists():
        original.write_bytes(config.read_bytes())
    cfg = yaml.safe_load(original.read_text())
    cfg['runner']['max_steps'] = decision['steps']
    cfg['runner']['deadline_unix'] = decision['training_deadline_unix']
    cfg['actor']['optim'].update(total_training_steps=decision['steps'], lr_warmup_steps=decision['warmup_steps'])
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    decision['config_sha256'] = hashlib.sha256(config.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(decision, indent=2))
    return decision


def checkpoint_is_complete(actor: Path):
    try:
        step = int(actor.parent.name.removeprefix('global_step_'))
        marker = json.loads((actor.parent/'checkpoint_complete.json').read_text())
        weights = actor/'model_state_dict/full_weights.pt'
        return (marker['global_step'] == step and (actor/'dcp_checkpoint/.metadata').is_file()
                and weights.stat().st_size > 0 and weights.stat().st_size == marker['full_weights_bytes'])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def install_termination_handler():
    """Route controller SIGTERM through the child-cleanup exception path."""
    import signal
    def interrupted(signum, frame):
        raise InterruptedError('Controller received SIGTERM')
    signal.signal(signal.SIGTERM, interrupted)


def terminate_process_group(proc, grace_seconds=30):
    """Stop the owned session, including children that outlive its leader."""
    import os
    import signal
    import subprocess
    previous = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        end = time.monotonic() + grace_seconds
        while time.monotonic() < end:
            proc.poll()
            try:
                os.killpg(proc.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(min(.1, max(0, end-time.monotonic())))
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)
    finally:
        signal.signal(signal.SIGTERM, previous)
