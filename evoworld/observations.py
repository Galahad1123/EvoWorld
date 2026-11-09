"""用户确认的正式腕部相机，以及跨数据/权重的观测身份检查。"""
import json
from pathlib import Path
from .config import ROOT,load
CONTRACT='milk-bin-side70-v1'

def apply_wrist(camera,config=None):
 c=load() if config is None else config
 if c.get('observation_contract')!=CONTRACT:
  raise ValueError(f'正式任务必须使用 {CONTRACT}；旧中央视角数据/权重需重新生成')
 profile=json.loads((ROOT/'configs/wrist-side70.json').read_text())
 camera.offset.pos=tuple(profile['position_hand_m'])
 camera.offset.rot=tuple(profile['rotation_ros_wxyz'])
 camera.offset.convention='ros'
 return profile

def verify_artifact(root,name):
 filename='source-manifest.json' if name=='train_data' else 'observation-contract.json'
 data=json.loads((Path(root)/filename).read_text())
 if data.get('observation_contract')!=CONTRACT:
  raise ValueError(f'Artifact observation contract mismatch: {name}')
