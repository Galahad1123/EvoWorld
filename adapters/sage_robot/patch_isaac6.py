"""Mechanical SDK namespace compatibility on isolated copies; no task algorithm changes."""
import json,re
from pathlib import Path
root=Path(__file__).resolve().parents[2]
base=root/'adapters/sage_robot/src'
replacements={
 'from omni.isaac.kit import SimulationApp':'from isaacsim import SimulationApp',
 'omni.isaac.core.utils.nucleus':'isaacsim.storage.native',
 'omni.isaac.core.utils':'isaacsim.core.utils',
 'omni.isaac.core.simulation_context':'isaacsim.core.api.simulation_context',
 'omni.isaac.core.world':'isaacsim.core.api.world',
 'omni.isaac.core.materials':'isaacsim.core.api.materials',
 'omni.isaac.core.objects':'isaacsim.core.api.objects',
 'omni.isaac.cloner':'isaacsim.core.cloner',
 'omni.isaac.motion_generation':'isaacsim.robot_motion.motion_generation',
 'omni.isaac.version':'isaacsim.core.version',
}
classes={'Articulation':'SingleArticulation','ArticulationView':'Articulation',
 'GeometryPrim':'SingleGeometryPrim','RigidPrim':'SingleRigidPrim','RigidPrimView':'RigidPrim','XFormPrimView':'XFormPrim'}
changed=[]
for parent in [base/'IsaacLab',base/'server']:
 for p in parent.rglob('*.py'):
  old=p.read_text();new=old
  def imp(m):
   names=[n.strip() for n in m[1].split(',')]
   return 'from isaacsim.core.prims import '+', '.join(classes.get(n,n)+' as '+n if n in classes else n for n in names)
  new=re.sub(r'^from omni\.isaac\.core\.(?:prims|articulations) import ([A-Za-z0-9_, ]+)$',imp,new,flags=re.M)
  for a,b in replacements.items():new=new.replace(a,b)
  if p.name=='app_launcher.py':
   new=new.replace('found_modules += ["omni.isaac.kit.app_framework"]','found_modules += [x for x in ["omni.isaac.kit.app_framework"] if x in sys.modules]')
  if p.name=='assets.py' and 'NUCLEUS_ASSET_ROOT_DIR' in new:
   new=new.replace('carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")','os.environ.get("EVOWORLD_SAGE_ASSET_ROOT", carb.settings.get_settings().get("/persistent/isaac/asset_root/default"))')
  if old!=new:p.write_text(new);changed.append(str(p.relative_to(root)))
(root/'adapters/sage_robot/namespace-patch-files.json').write_text(json.dumps(changed,indent=2))
print('patched',len(changed),'files')
