"""Portable entry for the verified standalone LIBERO baseline; never starts training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evoworld.config import load

def command(config, run_id, stage, num_envs):
    if not run_id or run_id in ('.', '..') or Path(run_id).name != run_id:
        raise ValueError('run-id must be a single nonempty directory name')
    if num_envs < 1:
        raise ValueError('num-envs must be positive')
    source = config['upstreams']['RLinf']
    env = dict(os.environ)
    env.update(EVOWORLD_ROOT=config['storage_root'],
               EVOWORLD_RUN_ID=run_id,
               EVOWORLD_RL_OUTPUT_ROOT=config['research']['output_root'],
               EVOWORLD_RLINF_SOURCE=source,
               EVOWORLD_LIBERO_CHECKPOINT=config['resources']['libero_checkpoint'],
               PYTHONPATH=source + os.pathsep + env.get('PYTHONPATH', ''),
               HF_ENDPOINT=config.get('network', {}).get('hf_endpoint', 'https://hf-mirror.com'),
               HF_HOME=str(Path(config['paths']['cache'])/'huggingface'),
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', ROBOT_PLATFORM='LIBERO', LIBERO_TYPE='standard',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    argv = [config['environments']['libero'], str(Path(__file__).with_name('evaluate.py')),
            '--stage', stage, '--num-envs', str(num_envs)]
    return argv, env

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', choices=['env', 'smoke', 'full'], required=True)
    parser.add_argument('--num-envs', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = load(args.config)
    argv, env = command(config, args.run_id, args.stage, args.num_envs)
    if args.dry_run:
        print(json.dumps({'command':argv,'output_root':config['research']['output_root'],
                          'rlinf':config['upstreams']['RLinf']}, indent=2))
        return
    output = Path(config['research']['output_root'])/'runs/libero/pi05-sft'/args.run_id
    formal = output/'metrics/full-episodes.jsonl'
    if (output/'metrics/summary.json').exists() or (formal.exists() and formal.stat().st_size):
        raise RuntimeError('Existing formal results are immutable; choose a new run ID')
    for name in ('logs', 'provenance'):
        (output/name).mkdir(parents=True, exist_ok=True)
    def version(path):
        return {'commit': subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
                'status': subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=normal'],text=True)}
    record = {'command':argv, 'evoworld':version(ROOT),
              'rlinf':version(config['upstreams']['RLinf']),
              'checkpoint':config['resources']['libero_checkpoint']}
    (output/'provenance'/f'{args.stage}-launch.json').write_text(json.dumps(record,indent=2))
    with (output/'provenance'/f'{args.stage}-packages.txt').open('w') as handle:
        subprocess.run([argv[0],'-m','pip','freeze'],stdout=handle,stderr=subprocess.STDOUT,check=True)
    log = output/'logs'/f'{args.stage}.log'
    print(f'Log: {log}', flush=True)
    with log.open('a') as handle:
        code = subprocess.call(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
    (output/'provenance'/f'{args.stage}-exit.json').write_text(json.dumps({'exit_code':code}))
    raise SystemExit(code)

if __name__ == '__main__':
    main()
