"""Check a native saved SFT checkpoint against the downloaded base weights."""
from pathlib import Path
import argparse,json
import torch
from safetensors import safe_open
from rlinf.models.embodiment.openpi.pi0 import Pi0
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf_extension import freeze_sft_vlm
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[2]
with torch.device('meta'):model=Pi0(Pi0Config(pi05=True,action_horizon=20,discrete_state_input=True))
freeze_sft_vlm(model)
saved=torch.load(Path(a.checkpoint)/'model_state_dict/full_weights.pt',map_location='cpu',weights_only=True,mmap=True)
frozen=0;changed=[]
with safe_open(ROOT/'models/pi05-base-openpi/model.safetensors',framework='pt',device='cpu') as source:
 for name,param in model.named_parameters():
  reference=source.get_tensor(name).to(saved[name].dtype)
  assert torch.isfinite(saved[name]).all(),name
  if not param.requires_grad:
   assert torch.equal(reference,saved[name]),'Frozen parameter changed: '+name
   frozen+=param.numel()
  elif not torch.equal(reference,saved[name]):changed.append(name)
assert 'action_out_proj.weight' in changed
result={'passed':True,'frozen_parameters_exactly_unchanged':frozen,'changed_trainable_tensor_count':len(changed),'action_head_changed':True}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
