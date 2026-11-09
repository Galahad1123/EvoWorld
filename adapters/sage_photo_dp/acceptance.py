"""Physical acceptance of the exact expert interval saved in the HDF5."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def check_episode(run,attempt,end):
 run=Path(run)
 trace=[json.loads(x) for x in (run/'physical-trace.jsonl').read_text().splitlines()]
 rows=[r for r in trace if r['attempt']==attempt and 21<=r['iteration']<end]
 roots=[json.loads(x) for x in (run/'root-trace.jsonl').read_text().splitlines()]
 roots=[r for r in roots if r['attempt']==attempt and 21<=r['iteration']<end]
 if len(rows)!=end-21 or len(roots)!=len(rows) or len(rows)<25:raise ValueError('Incomplete trace interval')
 assert [r['iteration'] for r in rows]==[r['iteration'] for r in roots]
 z0=next(r['object'][2] for r in trace if r['attempt']==attempt and r['iteration']==20)
 states=np.array([r['object'] for r in rows]);root=np.array([r['root'] for r in roots])
 initial=json.loads((run/f'initial-{attempt:03}.json').read_text());goal=np.array(initial['place_world'])[0,:2]
 profile=json.loads((run/'scene-profile.json').read_text());milk=next(o for o in profile['objects'] if o['id']=='milk')
 desk=next(o for o in profile['objects'] if o['id']=='desk');table_z=desk['pose_world'][2]
 from pxr import Usd
 from collision_bridge import collect_colliders
 vertices,_,_=collect_colliders(Usd.Stage.Open(str(Path(profile['source_bundle'])/milk['asset'])),'/Asset',True)
 rot=Rotation.from_quat(root[-1,[4,5,6,3]]).as_matrix()
 bottom=(vertices@rot.T+root[-1,:3])[:,2].min()
 tilt=np.degrees(np.arccos(np.clip(rot[2,2],-1,1)))
 metrics={'lift_m':float(states[:,2].max()-z0),'xy_error_m':float(np.linalg.norm(root[-1,:2]-goal)),
 'support_error_m':float(bottom-table_z),'tilt_deg':float(tilt),
 'tail_angular_speed_rad_s':float(np.linalg.norm(states[-25:,10:13],axis=1).max()),
 'tail_speed_m_s':float(np.linalg.norm(states[-25:,7:10],axis=1).max()),
 'finger_opening_m':float(np.mean(rows[-1]['q'][-2:]))}
 checks={'lift':metrics['lift_m']>.03,'placement':metrics['xy_error_m']<.05,
 'support':abs(metrics['support_error_m'])<.015,'upright':metrics['tilt_deg']<15,
 'stable':metrics['tail_speed_m_s']<.05,'angular_stable':metrics['tail_angular_speed_rad_s']<.2,'released':metrics['finger_opening_m']>.025}
 return {'attempt':attempt,'frames':len(rows),'end_exclusive':end,'metrics':metrics,'checks':checks,'success':all(checks.values())}

if __name__=='__main__':
 import argparse,h5py
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run)
 mapping=[json.loads(x) for x in (run/'demo-attempts.jsonl').read_text().splitlines()]
 results=[]
 with h5py.File(next(run.glob('robomimic_data/*/experts.hdf5')),'r') as f:
  for row in mapping:
   name=f"demo_{row['demo']}";actions=f['data/'+name+'/actions'][:]
   result=check_episode(run,row['attempt'],21+len(actions))
   records=[json.loads(x) for x in (run/'physical-trace.jsonl').read_text().splitlines()]
   recorded=np.array([r['action'] for r in records if r['attempt']==row['attempt'] and 21<=r['iteration']<21+len(actions)])
   np.testing.assert_array_equal(recorded,actions)
   for key in ['rgb_tabletop','rgb_wrist']:
    obs=f['data/'+name+'/obs/'+key][:];nxt=f['data/'+name+'/next_obs/'+key][:]
    assert obs.shape==(len(actions),128,128,3) and obs.dtype==np.uint8
    np.testing.assert_array_equal(nxt[:-1],obs[1:])
   results.append({'demo':name,**result})
 report={'all_pass':bool(results) and all(r['success'] for r in results),'episodes':results,'successes':sum(r['success'] for r in results),'total':len(results)}
 (run/'photo-physical-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
 if not report['all_pass']:raise SystemExit(1)
