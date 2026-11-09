"""Compose native SFT config and load a real two-camera transformed batch."""
import argparse,json
from pathlib import Path
import torch
from hydra import initialize_config_dir,compose
from omegaconf import open_dict
from rlinf_extension import register
register()
from rlinf.data.datasets.openpi import build_openpi_sft_dataloader
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args()
with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent),version_base='1.1'):cfg=compose(config_name='sft')
with open_dict(cfg):
 cfg.data.train_data_paths=str(Path(a.dataset).resolve())
 cfg.actor.model.openpi_data.norm_stats_path=str(Path(a.dataset).resolve()/'norm_stats.json')
 cfg.data.num_workers=0
loader,_=build_openpi_sft_dataloader(cfg,1,0,cfg.data.train_data_paths)
obs,actions=next(iter(loader))
report={'actions':list(actions.shape),'state':list(obs.state.shape),'images':{k:list(v.shape) for k,v in obs.images.items()},'image_masks':{k:v.tolist() for k,v in obs.image_masks.items()},'tokens':list(obs.tokenized_prompt.shape)}
assert actions.shape==(cfg.actor.micro_batch_size,20,32), report
assert torch.isfinite(actions).all()
assert obs.image_masks['base_0_rgb'].all() and obs.image_masks['left_wrist_0_rgb'].all()
assert not obs.image_masks['right_wrist_0_rgb'].any()
print(json.dumps(report,indent=2))
(Path(a.dataset)/'sft-batch-verified.json').write_text(json.dumps(report,indent=2))
