"""Read-only checks for native RLinf DSRL; never changes a training update."""
import hashlib
import torch

HEADS=('dsrl_action_noise_net','actor_image_encoder','actor_state_encoder','critic_image_encoder','critic_state_encoder','q_head')

def canonical(name):return name.replace('_fsdp_wrapped_module.','')

def digest_named(items):
 h=hashlib.sha256()
 for name,value in sorted(items):
  value=value.detach().cpu().contiguous()
  h.update(name.encode());h.update(str((value.dtype,tuple(value.shape))).encode())
  h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
 return h.hexdigest()

def parameter_report(model):
 groups={'frozen':[],'actor':[],'critic':[]};counts={'frozen':0,'trainable':0}
 for raw,p in model.named_parameters():
  name=canonical(raw);head=name.split('.')[0]
  if p.requires_grad:assert head in HEADS, f'Unexpected trainable decoder parameter: {name}'
  if head not in HEADS:
   assert not p.requires_grad,name
   groups['frozen'].append((name,p));counts['frozen']+=p.numel()
  else:
   key='critic' if head.startswith('critic_') or head=='q_head' else 'actor'
   groups[key].append((name,p));counts['trainable']+=p.numel() if p.requires_grad else 0
 return {**{k:digest_named(v) for k,v in groups.items()},'parameters':counts}

def assert_progress(before,after):
 assert before['frozen']==after['frozen'],'Frozen decoder weights changed'
 assert before['actor']!=after['actor'],'Noise actor did not update'
 assert before['critic']!=after['critic'],'Critic did not update'

def digest_tree(value):
 items=[]
 def visit(obj,path):
  if torch.is_tensor(obj):items.append((path,obj))
  elif isinstance(obj,dict):
   for k,v in sorted(obj.items(),key=lambda x:str(x[0])):visit(v,path+'/'+str(k))
  elif isinstance(obj,(list,tuple)):
   for k,v in enumerate(obj):visit(v,path+'/'+str(k))
  else:items.append((path,torch.tensor(list(repr(obj).encode()),dtype=torch.uint8)))
 visit(value,'');return digest_named(items)

def worker_report(worker):
 cache=worker.replay_buffer._flat_trajectory_cache
 replay={str(i):cache.get(i) for i in sorted(cache.cache)}
 return {'model':parameter_report(worker.model),'target':parameter_report(worker.target_model),
         'actor_optimizer':digest_tree(worker.optimizer.state_dict()),'critic_optimizer':digest_tree(worker.qf_optimizer.state_dict()),
         'alpha_optimizer':digest_tree(worker.alpha_optimizer.state_dict()),'alpha':digest_tree(worker.entropy_temp.state_dict()),
         'target_shadow':digest_named((canonical(n),v) for n,v in worker._target_shadow_f32.items()),
         'replay_contents':digest_tree(replay),
         'replay_rng':digest_tree(worker.replay_buffer.random_generator.get_state()),
         'torch_rng':digest_tree(torch.get_rng_state()),
         'cuda_rng':digest_tree(torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []),
         'schedulers':digest_tree([worker.lr_scheduler.state_dict(),worker.qf_lr_scheduler.state_dict(),worker.alpha_lr_scheduler.state_dict()]),
         'update_step':worker.update_step,'replay_size':len(worker.replay_buffer)}
