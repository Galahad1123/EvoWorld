"""CPU-only review of recorded Phase 3 attempts; never changes original simulator reports."""
import contextlib,hashlib,io,json,re,sys
from pathlib import Path
import h5py,numpy as np
from read_trajectory import read
root=Path(__file__).resolve().parents[2]
run=root/'experiments/phase3-franka-20261107-203852-xot5d9lo'
episode=run/'episode';dataset=episode/'trajectory.hdf5'
readback=read(dataset);(episode/'readback.json').write_text(json.dumps(readback,indent=2))
original=root/'upstream/sage/server/isaaclab/read_pick_n_place_data.py'
src,count=re.subn(r'^file_path = .*$',lambda _: 'file_path = '+repr(str(dataset)),original.read_text(),count=1,flags=re.M)
assert count==1
stream=io.StringIO()
with contextlib.redirect_stdout(stream):exec(compile(src,str(original),'exec'),{})
(run/'upstream-reader.log').write_text(stream.getvalue())
assert 'Total number of episodes: 1' in stream.getvalue()
with h5py.File(dataset) as f:
    d=f['data/demo_0'];pos=np.concatenate((d['obs/object_pos'][:1],d['next_obs/object_pos'][:]))
    vel=np.concatenate((d['obs/object_vel'][:1],d['next_obs/object_vel'][:]))
    tail=pos[-61:];speed=np.linalg.norm(vel[-61:],axis=1)
    diagnostics={'tail_center_span_xyz_m':np.ptp(tail,axis=0).tolist(),
        'tail_max_position_step_m':float(np.linalg.norm(np.diff(tail,axis=0),axis=1).max()),
        'tail_mean_speed_m_s':float(speed.mean()),'tail_samples_over_velocity_limit':int((speed>=.03).sum()),
        'joint_motion_range':np.ptp(d['next_obs/joint_pos'][:],axis=0).tolist()}
attempts=[]
for p in sorted((root/'experiments').glob('phase3-franka-*')):
    report=json.loads((p/'report.json').read_text())
    attempts.append({'run':p.name,'passed':report['passed'],'wall_seconds':report['wall_seconds'],
        'whole_gpu_peak_mib':report['whole_gpu_peak_mib'],'metrics':report['episode_report'].get('metrics'),
        'startup_error':(p/'episode/error.json').exists()})
result={'chain_readback_passed':True,'strict_episode_success':False,'phase3_status':'partial: compatibility task pipeline demonstrated; strict settling gate remains failed',
    'run':run.name,'readback':readback,'diagnostics':diagnostics,'attempts':attempts,
    'total_attempt_wall_seconds':sum(a['wall_seconds'] for a in attempts),
    'new_gpt_calls':0,'new_gpu_runs_in_this_review':0,
    'official_reader_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
    'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
    'limitations':['Original SAGE IsaacLab/M2T2/cuRobo pipeline not reproduced','No training consumer or RL policy acceptance',
                   'End-state velocity gate failed; no success-rate claim','SDF contact jitter is a hypothesis, not isolated root cause',
                   'Task robot base mounted/fixed in world; mounting hardware not modeled','No new GPT task planning: deterministic controller baseline'],
    'review_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(run/'reviewed-report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
