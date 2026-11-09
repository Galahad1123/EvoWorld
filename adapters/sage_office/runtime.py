"""Explicit office option layered on the accepted photo task; no default override."""
import copy,json
from pathlib import Path

def configure(cfg,background,tabletop_profile=None):
 from omni.isaac.lab.assets import AssetBaseCfg
 import omni.isaac.lab.sim as sim
 from omni.isaac.lab.sensors import CameraCfg
 background=Path(background).resolve()
 manifest=json.loads((background/'manifest.json').read_text());assert manifest['complete']
 removed=[]
 for key in list(vars(cfg.scene)):
  if key.startswith(('room_','floor_','wall_','window_','door_')) and getattr(cfg.scene,key) is not None:
   setattr(cfg.scene,key,None);removed.append(key)
 cfg.scene.office_background=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/OfficeBackground',spawn=sim.UsdFileCfg(usd_path=str(background/'office.usdz')))
 cfg.scene.light.spawn.color=(1.,1.,1.);cfg.scene.light.spawn.intensity=1500.
 cfg.scene.camera_office_overview=CameraCfg(prim_path='{ENV_REGEX_NS}/camera_office_overview',width=960,height=720,data_types=['rgb'],spawn=sim.PinholeCameraCfg(focal_length=20.,horizontal_aperture=24.,clipping_range=(.01,100.)))
 settings={'background_manifest':manifest,'removed_previous_scene_entries':removed,'lighting':'white dome, intensity1500','overview_position':[.15,.25,2.45],'overview_target':[2.45,2.15,.95]}

 if tabletop_profile is not None:
  profile=json.loads(Path(tabletop_profile).read_text())
  assert len(profile['position_world_m'])==3 and len(profile['target_world_m'])==3
  settings['tabletop_camera']=profile
  if 'focal_length_mm' in profile:
   assert profile['focal_length_mm']>0
   cfg.scene.camera_tabletop.spawn.focal_length=float(profile['focal_length_mm'])
  cfg.scene.camera_tabletop_preview=copy.deepcopy(cfg.scene.camera_tabletop)
  cfg.scene.camera_tabletop_preview.prim_path='{ENV_REGEX_NS}/camera_tabletop_preview'
  cfg.scene.camera_tabletop_preview.width=768;cfg.scene.camera_tabletop_preview.height=768
 return settings

def attach(env,settings):
 import torch
 env.scene['camera_office_overview'].set_world_poses_from_view(torch.tensor([settings['overview_position']],device=env.device),torch.tensor([settings['overview_target']],device=env.device))

 if 'tabletop_camera' in settings:
  profile=settings['tabletop_camera']
  for name in ['camera_tabletop','camera_tabletop_preview']:
   env.scene[name].set_world_poses_from_view(torch.tensor([profile['position_world_m']],device=env.device),torch.tensor([profile['target_world_m']],device=env.device))
