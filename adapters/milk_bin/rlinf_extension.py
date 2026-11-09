"""RLinf extension for EvoWorld: task transforms and SFT-only VLM freezing."""
import dataclasses

CONFIG_NAME='pi05_evoworld_milk_bin'
TASK='Pick up the oat milk carton and place it inside the blue storage box.'

def freeze_sft_vlm(model):
 model.freeze_vlm()
 import torch
 keep_float32=('img.stem.', 'img.pos_embedding', 'pre_attention_norms', 'pre_ffw_norms', 'final_norms')
 for name,p in model.named_parameters():
  dtype=torch.float32 if p.requires_grad or any(x in name for x in keep_float32) else torch.bfloat16
  p.data=p.data.to(dtype)
 report={'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
         'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad)}
 model.evoworld_freeze_report=report
 return report

def build_model(cfg,torch_dtype=None):
 import os,torch
 if os.environ.get('EVOWORLD_MODEL_GPU_FRACTION'):
  torch.cuda.set_per_process_memory_fraction(float(os.environ['EVOWORLD_MODEL_GPU_FRACTION']))
 from rlinf.models.embodiment.openpi import get_model
 model=get_model(cfg,torch_dtype)
 if cfg.openpi.config_name==CONFIG_NAME and cfg.openpi.task=='sft':freeze_sft_vlm(model)
 return model

def register():
 from openpi.models.pi0_config import Pi0Config
 from openpi.training.config import DataConfig,AssetsConfig,TrainConfig
 from rlinf.models.embodiment.openpi.dataconfig import _CONFIGS_DICT
 from rlinf.models.embodiment.openpi.dataconfig.isaaclab_dataconfig import LeRobotIsaacLabStackCubeDataConfig
 from rlinf.models import register_model
 _CONFIGS_DICT[CONFIG_NAME]=TrainConfig(
  name=CONFIG_NAME,model=Pi0Config(pi05=True,action_horizon=20,discrete_state_input=True),
  data=LeRobotIsaacLabStackCubeDataConfig(repo_id='evoworld/milk_bin_train',
   base_config=DataConfig(prompt_from_task=False),assets=AssetsConfig(asset_id='milk_bin'),default_prompt=TASK),
  num_workers=4)
 register_model('openpi',build_model,force=True)
 from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS
 from rlinf_env import MilkBinEnv
 REGISTER_ISAACLAB_ENVS['EvoWorld-Milk-Bin-Franka-v0']=MilkBinEnv
