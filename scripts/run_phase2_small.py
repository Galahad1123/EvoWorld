"""Two fixed-budget sequential generation jobs; retain failures, never replace them."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GROUPS=[
 {'name':'central_4x4','width':4.0,'length':4.0,'yaw':0,'opposite_axis':'y',
  'prompt':'Generate a NEW 4.0 by 4.0 meter single meeting room, height 2.7 m. Place exactly one table near the room center with yaw 0, two chairs on opposite north and south sides facing the table, and one mug near the tabletop center. Choose the positions yourself. Keep safe separation and use only the supplied asset catalog. No doors/windows or extra objects.'},
 {'name':'rotated_4p5x3p5','width':4.5,'length':3.5,'yaw':90,'opposite_axis':'x',
  'prompt':'Generate a NEW 4.5 by 3.5 meter single meeting room, height 2.7 m. Place exactly one table near the room center with yaw 90 degrees, two chairs on opposite east and west sides facing the table, and one mug near the tabletop center. Choose the positions yourself. Keep safe separation and use only the supplied asset catalog. No doors/windows or extra objects.'},
]

def gpu_sample():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,check=True)
        memory,util=(float(x.strip()) for x in r.stdout.strip().split(','))
        return {'memory_mib':memory,'utilization_percent':util}
    except Exception:
        return {'measurement_unavailable':True}

def summarize_run(run,group,row):
    row=dict(row)
    request_files=list(run.glob('request-*.json'));response_files=list(run.glob('response-*.json'))
    responses=[];corrupt=[]
    for path in response_files:
        try:
            value=json.loads(path.read_text())
            if not isinstance(value.get('usage'),dict): raise ValueError('Missing usage')
            responses.append(value)
        except (ValueError,TypeError,AttributeError): corrupt.append(path.name)
    row.update(request_records=len(request_files),completed_responses=len(responses),
        usage_complete=not corrupt and len(request_files)==len(responses),
        input_tokens=sum(x['usage'].get('input_tokens',0) for x in responses),
        output_tokens=sum(x['usage'].get('output_tokens',0) for x in responses))
    if corrupt:
        row.update(passed=False,artifact_error='Incomplete response artifacts: '+','.join(corrupt))
        return row
    try:
        if (run/'summary.json').exists():
            summary=json.loads((run/'summary.json').read_text());scene=run/summary['completion']['scene_id']
            plan=json.loads((scene/'plan.json').read_text());layout=json.loads((scene/'layout.json').read_text())
            table=next(o for o in layout['rooms'][0]['objects'] if o['type']=='table')
            chairs=[o for o in layout['rooms'][0]['objects'] if o['type']=='chair']
            axis=group['opposite_axis']
            semantic=plan['width']==group['width'] and plan['length']==group['length'] and table['rotation']['z']==group['yaw'] and (chairs[0]['position'][axis]-table['position'][axis])*(chairs[1]['position'][axis]-table['position'][axis])<0
            physics=json.loads((scene/'physics.json').read_text());render=json.loads((scene/'render.json').read_text())
            row.update(scene_id=scene.name,requested_layout_verified=semantic,physics_seconds=physics['timing']['simulation_seconds'],physics_steps=physics['timing']['observed_steps'],
                physics_wall_seconds=physics['timing']['wall_seconds'],motion_observed=physics['motion_check']['motion_observed'],tail_motion_bounded=physics['motion_check']['tail_motion_bounded'],
                max_trajectory_displacement_m=max(x['max_reference_center_displacement_m'] for x in physics['whole_trajectory'].values()),
                max_trajectory_rotation_degrees=max(x['max_rotation_change_degrees'] for x in physics['whole_trajectory'].values()),
                mug_table_gap_m=physics['mug_table_vertical_gap_m'],
                passed=bool(row.get('returncode')==0 and not row.get('timed_out') and summary['passed'] and physics['passed'] and render['passed'] and semantic))
    except (ValueError,KeyError,TypeError,StopIteration,OSError,AttributeError) as exc:
        row.update(passed=False,artifact_error=type(exc).__name__)
    return row

def main():
    batch=Path(tempfile.mkdtemp(prefix='phase2-small-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-',dir=ROOT/'experiments'))
    policy={'groups':2,'physics_steps_per_group':600,'seconds_per_group':10,'max_gpt_calls_per_group':6,
            'simulation_attempts_per_group':1,'wall_limit_seconds_per_group':360,'replacements':0,'gpu_sample_interval_seconds':2}
    (batch/'protocol.json').write_text(json.dumps({'policy':policy,'groups':GROUPS},indent=2))
    rows=[];baseline=gpu_sample();batch_start=time.monotonic()
    print('BATCH_DIR='+str(batch),flush=True)
    for group in GROUPS:
        parent=batch/group['name'];parent.mkdir()
        command=[str(ROOT/'envs/sage/bin/python'),'-u',str(ROOT/'adapters/minimal_sage/generate.py'),
                 '--physics-steps','600','--max-turns','6','--max-simulations','1','--run-parent',str(parent),'--prompt',group['prompt']]
        samples=[];started=time.monotonic();timed_out=False
        with (parent/'driver.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            try:
                while child.poll() is None:
                    elapsed=time.monotonic()-started
                    samples.append({'elapsed_seconds':elapsed,**gpu_sample()})
                    if elapsed>policy['wall_limit_seconds_per_group']:
                        timed_out=True
                        os.killpg(child.pid,signal.SIGTERM)
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: pass
                        # Reap any surviving descendants in this job's own process group.
                        try: os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError: pass
                        child.wait();break
                    time.sleep(2)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
        (parent/'gpu-samples.json').write_text(json.dumps(samples,indent=2))
        runs=list(parent.glob('sage-generated-*'))
        row={'group':group['name'],'returncode':child.returncode,'timed_out':timed_out,'wall_seconds':time.monotonic()-started,'passed':False,
             'gpu_peak_mib':max((x['memory_mib'] for x in samples if 'memory_mib' in x),default=None),'run_dir':str(runs[0]) if len(runs)==1 else None}
        if len(runs)==1:
            row=summarize_run(runs[0],group,row)
        rows.append(row)
        report={'policy':policy,'baseline_gpu':baseline,'results':rows,'batch_wall_seconds':time.monotonic()-batch_start,
                'completed_groups':len(rows),'passed_groups':sum(x['passed'] for x in rows),'scope':'Two asset-reuse layouts, 10 seconds of measured fixed-step physics each; not a statistical success-rate estimate'}
        (batch/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)
    print('PHASE2_SMALL_BATCH_COMPLETE='+str(batch),flush=True)
    return 0 if all(x['passed'] for x in rows) else 1

if __name__=='__main__': sys.exit(main())
