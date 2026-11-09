import argparse,json
from pathlib import Path
import h5py
import numpy as np
from metrics import success_checks,initial_com_height

def main():
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();run=Path(args.run)
    report=json.loads((run/'dataset-report.json').read_text());cache={};results={}
    with h5py.File(run/'training.hdf5','r') as f:
        for demo,meta in report['episodes'].items():
            src=Path(meta['source_run'])
            if str(src) not in cache:cache[str(src)]=[json.loads(x) for x in (src/'physical-trace.jsonl').read_text().splitlines()]
            n=meta['frames'];rows=[r for r in cache[str(src)] if r['attempt']==meta['attempt'] and 21<=r['iteration']<21+n]
            if len(rows)!=n:raise ValueError('Incomplete trace')
            np.testing.assert_array_equal(np.array([r['action'] for r in rows]),f[f'data/{demo}/actions'][:])
            init=json.loads((src/f"initial-{meta['attempt']:03}.json").read_text());z0=initial_com_height(cache[str(src)],meta['attempt'])
            states=np.array([r['object'] for r in rows]);goal=np.asarray(init['place_world'])[0,:2]
            checks=success_checks(states,z0,goal,rows[-1]['q'][-2:])
            results[demo]={'checks':checks,'success':all(checks.values()),'max_lift_m':float(states[:,2].max()-z0),'final_xy_error_m':float(np.linalg.norm(states[-1,:2]-goal)),'action_trace_exact':True}
    final={'episodes':results,'successes':sum(r['success'] for r in results.values()),'total':len(results),'all_pass':all(r['success'] for r in results.values())}
    (run/'expert-physical-verification.json').write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2))
    if not final['all_pass']:raise SystemExit('Physical acceptance failed; do not call these20 successful expert demos')
if __name__=='__main__':main()
