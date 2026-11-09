"""Audit commanded stationary intervals in the exact accepted HDF5 sequences."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run)
dt=json.loads((run/'scene-profile.json').read_text())['step_dt']
reports=[]
with h5py.File(next(run.glob('robomimic_data/*/experts.hdf5'))) as f:
 for name in f['data']:
  actions=f['data'][name]['actions'][:];n=len(actions);cuts=[0]
  for i in range(1,n):
   if np.max(np.abs(actions[i]-actions[i-1]))>1e-7:cuts.append(i)
  cuts.append(n)
  holds=[{'first_frame':b,'frames':e-b,'seconds':(e-b)*dt,'gripper':float(actions[b,-1])} for b,e in zip(cuts[:-1],cuts[1:]) if (e-b)*dt>=.3-1e-9]
  # Approach, descent, closure/lift, transport and final release must produce
  # separate stationary intervals; final release alone cannot satisfy this gate.
  assert len(holds)>=5,(name,holds)
  releases=np.flatnonzero((actions[1:,-1]>0)&(actions[:-1,-1]<0))+1
  assert len(releases)==1,(name,releases)
  r=int(releases[0]);pause=int(round(.3/dt))
  np.testing.assert_allclose(actions[r-pause:r,:7],np.tile(actions[r,:7],(pause,1)),atol=1e-7,rtol=0,err_msg='Release must follow a genuinely stationary 0.3s command interval')
  reports.append({'demo':name,'stationary_command_intervals':holds})
(run/'dwell-audit.json').write_text(json.dumps({'passed':True,'control_dt':dt,'minimum_command_hold_seconds':.3,'episodes':reports},indent=2))
print(json.dumps(reports,indent=2))
