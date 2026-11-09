"""Render the exact accepted dataset observations at the simulation's50Hz."""
import argparse,json
from pathlib import Path
import h5py,numpy as np,imageio.v2 as imageio
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('output');a=p.parse_args();run=Path(a.run)
with h5py.File(next(run.glob('robomimic_data/*/experts.hdf5'))) as f:
 g=f['data/demo_0'];n=len(g['actions'])
 with imageio.get_writer(a.output,fps=50) as writer:
  for i in range(n):writer.append_data(np.concatenate([g['obs/rgb_tabletop'][i],g['obs/rgb_wrist'][i]],axis=1))
print(json.dumps({'path':a.output,'frames':n,'fps':50,'seconds':n/50,'source':'accepted demo_0 current observations'}))
