"""Offline metric reassessment only: never starts GPT, MCP or Isaac."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'adapters/minimal_sage'))
sys.path.insert(0,str(ROOT/'assets/SAGE-10k/kits'))
from acceptance import assess_trace_motion
from tex_utils_local import dict_to_floor_plan,export_layout_to_mesh_dict_list_v2
from run_phase2_small import summarize_run

def main(batch):
    original=json.loads((batch/'report.json').read_text())
    protocol=json.loads((batch/'protocol.json').read_text())
    rows=[]
    for original_row,group in zip(original['results'],protocol['groups']):
        run=Path(original_row['run_dir']);row=summarize_run(run,group,original_row)
        if row.get('scene_id'):
            scene=run/row['scene_id'];path=scene/'physics.json'
            physics=json.loads(path.read_text());layout=json.loads((scene/'layout.json').read_text())
            meshes=export_layout_to_mesh_dict_list_v2(dict_to_floor_plan(layout),str(scene))
            centers={'/World/'+name:(np.asarray(data['mesh'].vertices).min(axis=0)+np.asarray(data['mesh'].vertices).max(axis=0))/2 for name,data in meshes.items() if not data['static']}
            motion=assess_trace_motion(physics['result']['traced_data_all'],expected_samples=601,tail_samples=61,reference_centers=centers)
            passed=bool(row['passed'] and motion['motion_observed'] and motion['tail_motion_bounded'])
            reviewed={'passed':passed,'original_physics_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'review_method':'CPU-only recalculation from unchanged recorded poses and official meshes; no additional simulation or API calls',
                'motion_check':motion,'tail_window_seconds':60/60,'analysis_sha256':hashlib.sha256((ROOT/'adapters/minimal_sage/acceptance.py').read_bytes()).hexdigest()}
            (scene/'physics-reviewed.json').write_text(json.dumps(reviewed,indent=2))
            row.update(passed=passed,center_tail_motion_bounded=motion['tail_motion_bounded'],tail_samples=61,tail_window_seconds=1.0,
                max_tail_center_step_m=max(v['max_tail_translation_step_m'] for v in motion['per_body'].values()))
        rows.append(row)
    report={**original,'results':rows,'passed_groups':sum(r['passed'] for r in rows),
        'offline_reassessment':True,'new_gpu_runs':0,'new_api_calls':0,
        'total_request_records':sum(r.get('request_records',0) for r in rows),'total_completed_responses':sum(r.get('completed_responses',0) for r in rows),
        'total_input_tokens':sum(r.get('input_tokens',0) for r in rows),'total_output_tokens':sum(r.get('output_tokens',0) for r in rows)}
    (batch/'reviewed-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if len(rows)==2 and all(r['passed'] for r in rows) else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('batch',type=Path)
    sys.exit(main(parser.parse_args().batch.resolve()))
