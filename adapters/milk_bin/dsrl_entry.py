"""Register DSRL hooks before Ray snapshots the native worker methods.

The native RLinf entrypoint, runner, workers and SAC loop remain in use.
Worker-process-only registration is too late for Ray's copied actor methods.
"""
from pathlib import Path
import runpy
from dsrl_extension import register

register()

if __name__ == '__main__':
 from evoworld.config import load
 entry = Path(load()['upstreams']['RLinf']) / 'examples/embodiment/train_embodied_agent.py'
 runpy.run_path(str(entry), run_name='__main__')
