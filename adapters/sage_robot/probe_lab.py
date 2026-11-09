"""Initialize SAGE's original Lab package on the installed SDK; no task replacement."""
import json,sys,traceback
from pathlib import Path
root=Path(__file__).resolve().parents[2];app=None
try:
 from omni.isaac.lab.app import AppLauncher
 launcher=AppLauncher(headless=True,enable_cameras=True,experience=str(root/'envs/isaacsim6/lib/python3.12/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit'))
 app=launcher.app
 import omni.isaac.lab.sim as sim_utils
 from omni.isaac.lab.assets import Articulation
 from omni.isaac.lab_assets.franka import FRANKA_PANDA_HIGH_PD_CFG
 print('ORIGINAL_LAB_IMPORT_OK',flush=True)
 sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.01,device='cuda:0'))
 print('ORIGINAL_LAB_CONTEXT_OK',flush=True)
except BaseException:
 traceback.print_exc()
 (root/'logs/phase3-original-lab-error.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2))
finally:
 if app:app.close()
