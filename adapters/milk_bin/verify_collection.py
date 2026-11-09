"""Recheck every saved expert interval with the current shared task predicate."""
import argparse,json,os
from pathlib import Path
import h5py,numpy as np
from milk_runtime import acceptance_result
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--count',type=int,required=True);a=p.parse_args()
run=Path(a.run).resolve();os.chdir(run)
mapping=[json.loads(x) for x in Path('demo-attempts.jsonl').read_text().splitlines()]
assert len(mapping)==a.count
trace=[json.loads(x) for x in Path('physical-trace.jsonl').read_text().splitlines()]
results=[]
with h5py.File(next(Path('.').glob('robomimic_data/*/experts.hdf5'))) as f:
 assert len(f['data'])==a.count
 for item in mapping:
  name=f"demo_{item['demo']}";g=f['data'][name];n=len(g['actions']);end=21+n
  result=acceptance_result(item['attempt'],end);assert result['success'],(name,result)
  actions=np.array([r['action'] for r in trace if r['attempt']==item['attempt'] and 21<=r['iteration']<end])
  np.testing.assert_array_equal(actions,g['actions'][:])
  for camera in ['rgb_tabletop','rgb_wrist']:
   obs=g['obs/'+camera];nxt=g['next_obs/'+camera]
   assert obs.shape==(n,224,224,3) and obs.dtype==np.uint8
   for start in range(0,n-1,64):
    stop=min(start+64,n-1)
    np.testing.assert_array_equal(nxt[start:stop],obs[start+1:stop+1])
  results.append({'demo':name,**result})
report={'passed':True,'count':len(results),'release_rule':'positive opening command and both measured fingers>39mm for entire final1s','episodes':results}
Path('collection-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps({'passed':True,'count':len(results)}))
