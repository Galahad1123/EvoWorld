"""One disposable native-model optimizer update; never a baseline checkpoint."""
from pathlib import Path
import argparse,json,time,hashlib
import torch
from hydra import initialize_config_dir,compose
from omegaconf import open_dict
from rlinf_extension import register,build_model
from rlinf.data.datasets.openpi import build_openpi_sft_dataloader
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args()
register();torch.cuda.set_per_process_memory_fraction(.55)
with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent),version_base='1.1'):cfg=compose(config_name='sft')
with open_dict(cfg):
 cfg.data.train_data_paths=str(Path(a.dataset).resolve());cfg.data.num_workers=0
 cfg.actor.model.openpi_data.norm_stats_path=str(Path(a.dataset).resolve()/'norm_stats.json')
 cfg.actor.model.precision='bf16'
loader,_=build_openpi_sft_dataloader(cfg,1,0,cfg.data.train_data_paths)
batch=next(iter(loader));start=time.monotonic()
model=build_model(cfg.actor.model).cuda();model.gradient_checkpointing_enable();model.train()
# The vision trunk has no optimizer parameters; explicitly retain eval mode.
model.img.eval()
def digest_frozen():
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  if not p.requires_grad:
   h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
before=digest_frozen();head_before=model.action_out_proj.weight.detach().clone()
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.95))
torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.monotonic()
loss=model(data=batch);assert torch.isfinite(loss);loss.backward()
assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
trainable=[p for p in model.parameters() if p.requires_grad]
assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in trainable)
assert all(torch.isfinite(p.grad).all() for p in trainable if p.grad is not None)
optimizer.step();torch.cuda.synchronize();step_seconds=time.monotonic()-begin
assert not torch.equal(model.action_out_proj.weight,head_before)
assert digest_frozen()==before
report={'passed':True,'purpose':'disposable one-update model smoke; not SFT training','precision':'native selective bf16 (formal SFT configuration remains fp32 master parameters)',**model.evoworld_freeze_report,'loss':float(loss.detach()),'optimizer_updates':1,'step_seconds':step_seconds,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'frozen_sha256_unchanged':before,'action_head_changed':True,'wall_seconds':time.monotonic()-start}
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
