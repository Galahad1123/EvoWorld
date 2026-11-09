"""Bounded task contract test in the RLinf interpreter, with native simulation."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
from sim_task import create_task
env,app=create_task(710,args.output)
import torch,numpy as np
try:
 obs,_=env.reset(seed=710)
 initial=env.env.scene['milk'].data.root_state_w.clone()
 for name in ['rgb_tabletop','rgb_wrist']:
  rgb=obs['policy'][name];assert rgb.shape==(1,224,224,3) and rgb.float().std()>5
 assert obs['policy']['state'].shape==(1,8)
 a=torch.tensor([[0.,0,0,0,0,0,1]],device=env.device)
 for _ in range(5):
  obs,reward,term,trunc,info=env.step(a)
  assert abs(float(reward[0])+.001)<1e-6 and not term.any() and not trunc.any()
 env.reset(seed=710)
 np.testing.assert_allclose(env.env.scene['milk'].data.root_state_w[:,:7].cpu(),initial[:,:7].cpu(),atol=2e-4)
 report={'passed':True,'steps':5,'reset_reproducible':True,'state_dim':8,'action_dim':7,'camera_shape':[1,224,224,3],'control_dt':env.env.step_dt}
 (Path(args.output).resolve()/'task-smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:env.close();app.close()
