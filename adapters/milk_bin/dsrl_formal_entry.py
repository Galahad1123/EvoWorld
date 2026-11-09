"""Native DSRL entry with verified worker hooks and a bounded wall-clock budget."""
import os,runpy
from pathlib import Path
from dsrl_extension import register
register()
from dsrl_budget import install_budget
from rlinf.runners.embodied_runner import EmbodiedRunner
install_budget(EmbodiedRunner,deadline=float(os.environ['EVOWORLD_DSRL_DEADLINE']))
if __name__=='__main__':
 runpy.run_path(str(Path(__import__('evoworld.config',fromlist=['load']).load()['upstreams']['RLinf'])/'examples/embodiment/train_embodied_agent.py'),run_name='__main__')
