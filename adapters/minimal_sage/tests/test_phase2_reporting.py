import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from run_phase2_small import summarize_run

def test_interrupted_response_keeps_failed_result_and_known_usage(tmp_path):
    (tmp_path/'request-01.json').write_text('{}')
    (tmp_path/'response-01.json').write_text('{"usage":')
    row=summarize_run(tmp_path,{}, {'passed':False})
    assert row['passed'] is False
    assert row['request_records']==1 and row['completed_responses']==0
    assert row['usage_complete'] is False
    assert row['artifact_error']

def test_truncated_summary_cannot_abort_batch_reporting(tmp_path):
    (tmp_path/'summary.json').write_text('{')
    row=summarize_run(tmp_path,{}, {'passed':False})
    assert row['passed'] is False and row['artifact_error']
