"""CPU evidence verification with independent quaternion and position formulas."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(out):
    out=Path(out);report=json.loads((out/'report.json').read_text());run=out.parent
    assert report['complete'] and report['no_training'] and report['no_simulator']
    assert digest(run/'training.hdf5')==report['dataset_sha256']
    assert digest(report['checkpoint'])==report['checkpoint_sha256']
    for name,sha in report['source_sha256'].items():assert digest(out/'source'/name)==sha
    data=json.loads((run/'dataset-report.json').read_text())
    assert report['test_episodes']==data['test']
    totals={m:[] for m in report['methods']};count=0
    with h5py.File(run/'training.hdf5','r') as f:
        for demo in report['test_episodes']:
            with np.load(out/(demo+'-predictions.npz')) as saved:
                ref=f['data/'+demo+'/actions'][:].astype(float)
                np.testing.assert_array_equal(saved['reference_actions'],ref)
                np.testing.assert_array_equal(saved['hold_observed_pose'],f['data/'+demo+'/obs/ee_pose'][:])
                count+=len(ref)
                for mode in totals:
                    for seed in report['seeds']:
                        p=saved[f'{mode}_seed{seed}'].astype(float)
                        assert p.shape==ref.shape and np.isfinite(p).all()
                        distance=np.sqrt(np.sum((p[:,:3]-ref[:,:3])**2,axis=1))*100
                        q=p[:,3:7]/np.linalg.norm(p[:,3:7],axis=1)[:,None]
                        r=ref[:,3:7]/np.linalg.norm(ref[:,3:7],axis=1)[:,None]
                        q[np.sum(q*r,axis=1)<0]*=-1
                        # Quaternion chord distances: SO(3) angle =4 atan2(|q-r|,|q+r|).
                        angle=np.degrees(4*np.arctan2(np.linalg.norm(q-r,axis=1),np.linalg.norm(q+r,axis=1)))
                        correct=(p[:,7]>=0)==(ref[:,7]>=0)
                        totals[mode].append((distance,angle,correct))
                        rec=next(x for x in report['episodes'] if (x['demo'],x['mode'],x['seed'])==(demo,mode,seed))
                        np.testing.assert_allclose([distance.mean(),angle.mean(),correct.mean()],
                          [rec['all'][k] for k in ['position_mean_cm','rotation_mean_deg','gripper_accuracy']],atol=1e-8)
    assert count==report['unique_frames']==400
    for mode,values in totals.items():
        d,a,c=[np.concatenate([v[i] for v in values]) for i in range(3)]
        r=report['aggregate'][mode]
        assert len(d)==r['frames']==count*len(report['seeds'])
        np.testing.assert_allclose([d.mean(),np.sqrt(np.mean(d*d)),np.percentile(d,95),a.mean(),np.percentile(a,95),c.mean()],
          [r[k] for k in ['position_mean_cm','position_rmse_cm','position_p95_cm','rotation_mean_deg','rotation_p95_deg','gripper_accuracy']],atol=1e-8)
    with (out/'per-frame-errors.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==count*len(report['seeds'])*len(totals)
    for row in rows:
        mode=row['mode'];seed=int(row['seed']);t=int(row['frame']);demo=row['demo']
        assert mode in totals and seed in report['seeds'] and demo in report['test_episodes']
        assert int(row['chunk_offset'])==(t%8 if mode=='chunk8' else 0)
        assert np.isclose(float(row['time_s']),t/data['fps'])
    assert len({(r['demo'],r['mode'],r['seed'],r['frame']) for r in rows})==len(rows)
    result={'passed':True,'unique_expert_frames':count,'scored_predictions':len(rows),
            'checks':['dataset and checkpoint immutable hashes','executed source snapshots',
                      'saved reference actions exactly equal original HDF5','independent position and rotation recomputation',
                      'per episode and aggregate metrics','CSV cardinality, timestamps and chunk offsets'],
            'verifier_sha256':digest(__file__)}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':verify(sys.argv[1])
