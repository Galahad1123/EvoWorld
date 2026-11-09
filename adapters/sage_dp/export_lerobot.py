"""SAGE HDF5 -> official LeRobot v3 writer; run in envs/sage-lerobot (CPU).

Preserves current observations, actions, rewards and original done flags. RGB is
H264; quantized depth and sampled points remain exact numeric arrays. next_obs
stays in the immutable HDF5 source (including final next_obs). No Hub upload.
"""
import argparse,hashlib,json,time
from pathlib import Path
import h5py
import numpy as np


def source_schema(source):
    names=sorted(source['data'],key=lambda n:int(n.split('_')[-1]))
    first=source['data'][names[0]];features={};mapping={}
    paths=['actions']+['obs/'+k for k in first['obs']]+[k for k in ['rewards','dones'] if k in first]
    for key in paths:
        x=first[key];shape=tuple(x.shape[1:]) or (1,)
        if key=='actions':target='action'
        elif key=='obs/joint_qpos':target='observation.state'
        elif key.startswith('obs/rgb_'):target='observation.images.'+key.split('/')[1]
        elif key.startswith('obs/'):target='observation.'+key.split('/')[1]
        else:target={'rewards':'next.reward','dones':'next.done'}[key]
        visual=key.startswith('obs/rgb_')
        features[target]={'dtype':'video' if visual else x.dtype.name,'shape':shape,'names':['height','width','channels'] if visual else None}
        mapping[target]=key
    if first['actions'].shape[1:]!=(8,) or 'obs/joint_qpos' not in first:raise ValueError('Expected SAGE absolute8D action and joint_qpos')
    features['action']['names']=['x_m','y_m','z_m','qw','qx','qy','qz','gripper_command']
    features['observation.state']['names']=[f'arm_joint_{i}_rad' for i in range(1,8)]+['gripper_command']
    for name in names:
        g=source['data'][name];n=len(g['actions'])
        if n==0:raise ValueError('Empty episode')
        if set(g['obs'])!=set(first['obs']):raise ValueError('Observation keys differ across episodes')
        for target,key in mapping.items():
            x=g[key]
            if len(x)!=n or (tuple(x.shape[1:]) or (1,))!=features[target]['shape']:raise ValueError('Inconsistent episode features')
            if x.dtype.name!=first[key].dtype.name or not np.isfinite(x[:]).all():raise ValueError('Nonfinite or inconsistent dtype')
    return names,features,mapping


def export_dataset(hdf5, output, repo_id, fps, task):
    if fps<=0 or not isinstance(fps,int):raise ValueError('Positive integer FPS required')
    output=Path(output).resolve();hdf5=Path(hdf5).resolve()
    if output.exists():raise FileExistsError(output)
    if not task.strip():raise ValueError('A task instruction is required')
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    start=time.monotonic()
    with h5py.File(hdf5,'r') as source:
        names,features,mapping=source_schema(source)
        dataset=LeRobotDataset.create(repo_id=repo_id,root=output,fps=fps,robot_type='franka',features=features,use_videos=True,video_backend='pyav',vcodec='h264',image_writer_threads=4)
        episodes=[];offset=0
        try:
            for index,name in enumerate(names):
                g=source['data'][name];n=len(g['actions'])
                for t in range(n):
                    frame={k:np.asarray(g[path][t]).reshape(features[k]['shape']) for k,path in mapping.items()}
                    frame['task']=task;dataset.add_frame(frame)
                dataset.save_episode()
                episodes.append({'index':index,'source_demo':name,'start':offset,'frames':n});offset+=n
                print(json.dumps({'exported_episode':index,'frames':n}),flush=True)
        finally:
            dataset.finalize();dataset.stop_image_writer()
        manifest={'source_hdf5':str(hdf5),'source_sha256':hashlib.sha256(hdf5.read_bytes()).hexdigest(),
                  'repo_id':repo_id,'fps':fps,'task':task,'mapping':mapping,'episodes':episodes,
                  'splits':{k:[x.decode() for x in source['mask'][k][:]] for k in source.get('mask',{})},
                  'source_env_args':str(source['data'].attrs.get('env_args','')),
                  'action_contract':'absolute panda_hand xyz metres in robot base, scalar-first quaternion, +1 open/-1 close',
                  'state_contract':'7 absolute arm joint angles plus gripper COMMAND; not an eighth joint angle',
                  'depth_contract':'original uint8 clipped/quantized depth; not metric depth',
                  'omitted_groups':['next_obs (retained in source HDF5, including last transition)'],
                  'done_contract':'copied original flags, all false in SAGE fixture; episode boundaries are metadata, not inferred RL terminations',
                  'image_encoding':'H264 via official writer, lossy RGB; numeric arrays exact',
                  'wall_seconds':time.monotonic()-start,'complete':True}
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
    write_split_configs(output)
    return manifest


def write_split_configs(output):
    """Official LeRobot defaults to all episodes. Supply explicit DatasetConfig fragments."""
    output=Path(output).resolve();m=json.loads((output/'source-manifest.json').read_text())
    indices={e['source_demo']:e['index'] for e in m['episodes']};splits={};seen=set()
    for name in ['train','valid','test']:
        if name not in m['splits']:continue
        values=[indices[demo] for demo in m['splits'][name]]
        if len(set(values))!=len(values) or seen.intersection(values):raise ValueError('Overlapping train/valid/test episodes')
        seen.update(values);splits[name]=values
    for name,values in splits.items():
        (output/(name+'-dataset-config.json')).write_text(json.dumps({'repo_id':m['repo_id'],'root':str(output),'episodes':values},indent=2))
    (output/'split-indices.json').write_text(json.dumps(splits,indent=2))
    return splits


def verify_dataset(output):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    output=Path(output);m=json.loads((output/'source-manifest.json').read_text())
    source=Path(m['source_hdf5'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=m['source_sha256']:raise ValueError('Source HDF5 changed')
    ds=LeRobotDataset(repo_id=m['repo_id'],root=output,video_backend='pyav',download_videos=False)
    data=ds.hf_dataset.with_format('numpy');image_errors=[]
    with h5py.File(source,'r') as f:
        for ep in m['episodes']:
            start=ep['start'];n=ep['frames'];g=f['data'][ep['source_demo']]
            numeric=[k for k in m['mapping'] if not k.startswith('observation.images.')]
            block=data.select_columns(numeric)[start:start+n]
            for key in numeric:
                expected=g[m['mapping'][key]][:];got=np.asarray(block[key]).reshape(expected.shape)
                np.testing.assert_array_equal(got,expected,err_msg=f'{ep["source_demo"]}/{key}')
            for t in sorted({0,n//2,n-1}):
                frame=ds[start+t]
                assert int(frame['episode_index'])==ep['index']
                assert abs(float(frame['timestamp'])-t/m['fps'])<1e-4
                for key,path in m['mapping'].items():
                    if key.startswith('observation.images.'):
                        rgb=frame[key].numpy().transpose(1,2,0)*255;expected=g[path][t]
                        mae=float(np.abs(rgb-expected).mean());image_errors.append(mae)
                        if mae>12:raise ValueError(f'Video frame alignment/color error: {ep}/{key}/{t}: {mae}')
    chunks=LeRobotDataset(repo_id=m['repo_id'],root=output,video_backend='pyav',download_videos=False,delta_timestamps={'action':[i/m['fps'] for i in range(16)]})
    with h5py.File(source,'r') as f:
        for ep in m['episodes']:
            frame=chunks[ep['start']+ep['frames']-1]
            expected=np.repeat(f['data'][ep['source_demo']]['actions'][-1][None],16,axis=0)
            np.testing.assert_array_equal(frame['action'].numpy(),expected)
            np.testing.assert_array_equal(frame['action_is_pad'].numpy(),np.r_[False,np.ones(15,dtype=bool)])
    splits=json.loads((output/'split-indices.json').read_text());counts={};seen=set()
    source_indices={e['source_demo']:e['index'] for e in m['episodes']}
    for name,indices in splits.items():
        assert indices==[source_indices[n] for n in m['splits'][name]]
        assert not seen.intersection(indices);seen.update(indices)
        filtered=LeRobotDataset(repo_id=m['repo_id'],root=output,episodes=indices,video_backend='pyav',download_videos=False)
        actual={int(x) for x in filtered.hf_dataset['episode_index']}
        assert actual==set(indices)
        counts[name]=len(actual)
    report={'episodes':len(m['episodes']),'frames':sum(e['frames'] for e in m['episodes']),'numeric_exact':True,
            'action_chunk_boundaries':True,'sampled_rgb_frames':len(image_errors),'max_sampled_rgb_mae':max(image_errors),
            'explicit_split_episode_counts':counts,'split_isolation_verified':True,
            'format_version':ds.meta.info['codebase_version'],'complete':True}
    (output/'roundtrip-report.json').write_text(json.dumps(report,indent=2));return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--hdf5');p.add_argument('--output',required=True);p.add_argument('--repo-id',default='local/evoworld-sage-cube');p.add_argument('--fps',type=int,default=50);p.add_argument('--task',default='Pick up the red cube and place it at the center of the table.');p.add_argument('--verify-only',action='store_true');p.add_argument('--write-split-configs',action='store_true');args=p.parse_args()
    if args.write_split_configs:write_split_configs(args.output)
    if not args.verify_only:
        if not args.hdf5:p.error('--hdf5 is required for export')
        export_dataset(args.hdf5,args.output,args.repo_id,args.fps,args.task)
    print(json.dumps(verify_dataset(args.output),indent=2))
if __name__=='__main__':main()
