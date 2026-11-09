"""Independent simulator rollout: held-out initialization, no expert actions in policy mode."""
import argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--checkpoint')
parser.add_argument('--mode',choices=['expert','policy'],required=True);parser.add_argument('--split',default='test')
parser.add_argument('--output',required=True)
from omni.isaac.lab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();app=AppLauncher(args).app
import hashlib,json,os,pickle,struct,subprocess,time,traceback
import numpy as np
import torch
import h5py
import imageio.v2 as imageio
from runtime import configure,attach,cpu
from metrics import success_checks
from omni.isaac.lab_tasks.manager_based.manipulation.lift_single_obj_scene.config.franka.ik_abs_env_cfg import FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser
from omni.isaac.lab.envs import ManagerBasedRLEnv

ROOT=Path(__file__).resolve().parents[2];run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
report=json.loads((run/'dataset-report.json').read_text());proc=None;status=1;results=[];started=time.monotonic()
def recv():
    import select
    if not select.select([proc.stdout],[],[],90)[0]:raise TimeoutError('Policy worker response timeout')
    head=proc.stdout.read(8)
    if len(head)!=8:raise RuntimeError('Policy worker exited; inspect worker.log')
    n=struct.unpack('!Q',head)[0];payload=proc.stdout.read(n)
    return pickle.loads(payload)
def request(msg):
    b=pickle.dumps(msg,protocol=5);proc.stdin.write(struct.pack('!Q',len(b)));proc.stdin.write(b);proc.stdin.flush();return recv()
try:
    config=next(run.glob('robomimic_data/*/params/env_init.yaml'))
    cfg=FrankaSingleObjSceneLiftEnvCfg_Vis_Teaser(config_yaml=str(config));configure(cfg,208)
    cfg.scene.num_envs=1;cfg.sim.device='cuda:0';cfg.terminations.time_out=None
    for key in list(vars(cfg.scene)):
        if 'camera_' in key and key not in {'camera_left','camera_right','camera_wrist'}:setattr(cfg.scene,key,None)
    for key in list(vars(cfg.observations.policy)):
        if '_vis' in key:setattr(cfg.observations.policy,key,None)
    env=ManagerBasedRLEnv(cfg=cfg);attach(env);robot=env.scene['robot']
    if args.mode=='policy':
        worker_env=dict(os.environ);worker_env.pop('PYTHONPATH',None)
        proc=subprocess.Popen([str(ROOT/'envs/sage-dp/bin/python'),str(ROOT/'adapters/sage_dp/policy_worker.py'),args.checkpoint],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(out/'worker.log').open('w'),env=worker_env)
        ready=recv();assert ready['ready'];obs_keys=ready['obs_keys']
    with h5py.File(run/'training.hdf5','r') as f:
        for demo in report[args.split]:
            meta=report['episodes'][demo];initial=json.loads((Path(meta['source_run'])/f"initial-{meta['attempt']:03}.json").read_text())
            env.reset(seed=208)
            tensor=lambda x:torch.as_tensor(x,device=env.device,dtype=torch.float32)
            robot.write_root_link_state_to_sim(tensor(initial['robot_root']))
            robot.write_joint_state_to_sim(tensor(initial['q']),tensor(initial['qd']));robot.reset()
            for name,values in initial['objects'].items():
                env.scene[name].write_root_state_to_sim(tensor(values));env.scene[name].reset()
            env.action_manager.reset();env.action_manager.process_action(tensor(initial['previous_action']))
            for name in ['left','right']:
                cam=initial['cameras'][name];env.scene['camera_'+name].set_world_poses(tensor(cam['position']),tensor(cam['quaternion']),convention='ros');env.scene['camera_'+name].reset()
            env.scene['camera_wrist'].reset();env.scene.write_data_to_sim();env.sim.forward();env.scene.update(env.physics_dt)
            # Camera pose changes and RTX textures need render-only warmup. Do not
            # advance physics or apply expert actions to prime a learned rollout.
            for _ in range(12):env.sim.render()
            for name in ['left','right','wrist']:env.scene['camera_'+name].update(0.0,force_recompute=True)
            obs=env.observation_manager.compute()['policy']
            # Raw collector starts after 21 settle steps. Restore that state, not the first expert target.
            init_error={k:float(np.max(np.abs(cpu(obs[k][0]).astype(float)-f[f'data/{demo}/obs/{k}'][0].astype(float)))) for k in ['ee_pose','joint_qpos','rgb_left','rgb_right']}
            image_mae={k:float(np.mean(np.abs(cpu(obs[k][0]).astype(float)-f[f'data/{demo}/obs/{k}'][0].astype(float)))) for k in ['rgb_left','rgb_right']}
            np.savez_compressed(out/(demo+'-initial-obs.npz'),**{k:cpu(obs[k][0]) for k in ['ee_pose','joint_qpos','rgb_left','rgb_right']})
            if any(init_error[k]>.001 for k in ['ee_pose','joint_qpos']):raise ValueError(f'Initial robot observation restore mismatch: {init_error}')
            if any(v>15 for v in image_mae.values()):raise ValueError(f'Initial visual observation restore mismatch: {image_mae}')
            if args.mode=='policy':request({'reset':True})
            obj=env.scene[initial['target']];z0=float(cpu(obj.data.root_com_state_w[0,2]));goal=np.asarray(initial['place_world'])[0,:2]
            horizon=len(f[f'data/{demo}/actions']) if args.mode=='expert' else 379
            states=[];actions=[];latencies=[];raw_actions=[];bad=None
            writer=imageio.get_writer(str(out/(demo+'.mp4')),fps=25)
            for t in range(horizon+50):
                if t<horizon:
                    begin=time.monotonic()
                    if args.mode=='expert':a=f[f'data/{demo}/actions'][t]
                    else:a=np.asarray(request({'obs':{k:cpu(obs[k][0]) for k in obs_keys}})['action'])
                    latencies.append(time.monotonic()-begin);raw_actions.append(a.copy())
                    if a.shape!=(8,) or not np.isfinite(a).all() or np.linalg.norm(a[3:7])<.01 or np.max(np.abs(a[:3]))>2:
                        bad='invalid policy action';break
                    a=a.astype('float32').copy();a[3:7]/=np.linalg.norm(a[3:7]);a[7]=1 if a[7]>=0 else -1
                obs=env.step(tensor(a)[None])[0]['policy']
                state=cpu(obj.data.root_com_state_w[0]).copy();states.append(state);actions.append(a.copy())
                if t%2==0:writer.append_data(np.concatenate([cpu(obs['rgb_left'][0]),cpu(obs['rgb_right'][0])],axis=1))
            writer.close()
            states=np.asarray(states);final=states[-1];tail=states[-25:]
            checks=success_checks(states,z0,goal,cpu(robot.data.joint_pos[0,-2:]))
            result={'demo':demo,'mode':args.mode,'steps':len(states),'checks':checks,'success':bad is None and all(checks.values()),
                    'error':bad,'max_lift_m':float(np.max(states[:,2])-z0),'final_xy_error_m':float(np.linalg.norm(final[:2]-goal)),
                    'initial_obs_max_abs_error':init_error,'initial_image_mae':image_mae,'policy_call_mean_s':float(np.mean(latencies)),
                    'initialization_source':str(Path(meta['source_run'])/f"initial-{meta['attempt']:03}.json")}
            np.savez_compressed(out/(demo+'.npz'),states=states,actions=np.asarray(actions),raw_actions=np.asarray(raw_actions),goal_xy=goal,initial_z=z0)
            results.append(result);print(json.dumps(result),flush=True)
    final={'mode':args.mode,'split':args.split,'episodes':results,'successes':sum(r['success'] for r in results),'total':len(results),'complete':True,'wall_seconds':time.monotonic()-started,
           'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
           'success_definition':'lift >3cm, final XY <8cm, open fingers >2.5cm, support Z within1.5cm, last0.5s linear speed <5cm/s',
           'scope':'held-out expert initialization; fixed room; 50 extra control steps holding last action'}
    (out/'report.json').write_text(json.dumps(final,indent=2));status=0;env.close()
except Exception:
    (out/'error.txt').write_text(traceback.format_exc());traceback.print_exc()
finally:
    if proc:
        proc.stdin.close()
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
    (out/'exit-status.txt').write_text(str(status));app.close()
raise SystemExit(status)
