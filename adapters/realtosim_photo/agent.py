"""Bounded GPT asset-authoring tool requests. Never executes code with credentials loaded."""
import base64
import json
from pathlib import Path
import sys
import time
from openai import OpenAI

run = Path(sys.argv[1]).resolve()
round_dir = run / sys.argv[2]
round_dir.mkdir(exist_ok=False)
prompt = Path(sys.argv[3]).read_text()
content = [{'type': 'input_text', 'text': prompt}]
paths = [run / 'input' / n for n in ['main.jpg', 'left.jpg', 'right.jpg']]
paths += [Path(p).resolve() for p in sys.argv[4:]]
for p in paths:
    mime = 'image/jpeg' if p.suffix.lower() in ['.jpg', '.jpeg'] else 'image/png'
    content += [{'type': 'input_text', 'text': f'Image: {p.name}'},
                {'type': 'input_image', 'image_url': f'data:{mime};base64,' + base64.b64encode(p.read_bytes()).decode(), 'detail': 'high'}]
tool = {'type': 'function', 'name': 'submit_scene_candidate',
    'description': 'Submit a complete auditable USD asset-authoring program and provenance route. The caller executes it without credentials, then returns real build/render/physics evidence for critique.',
    'parameters': {'type': 'object', 'properties': {'program': {'type': 'string'},
        'route_plan': {'type': 'object'}, 'assessment': {'type': 'object'}},
        'required': ['program', 'route_plan', 'assessment'], 'additionalProperties': False}}
import yaml
from evoworld.config import ROOT
gpt=yaml.safe_load((ROOT/'configs/gpt.yaml').read_text())
request = {'model': gpt['model'], 'reasoning': {'effort': gpt['reasoning_effort']}, 'max_output_tokens': gpt['max_output_tokens'],
           'tools': [tool], 'tool_choice': {'type': 'function', 'name': 'submit_scene_candidate'}}
(round_dir / 'request.json').write_text(json.dumps({**request, 'prompt': prompt, 'images': [str(p.relative_to(run)) for p in paths]}, indent=2))
started = time.monotonic()
try:
    with OpenAI(timeout=gpt['timeout_seconds'], max_retries=gpt['max_retries']) as client:
        with client.responses.create(**request, input=[{'role': 'user', 'content': content}], stream=True) as stream:
            events=0
            response=None
            with (round_dir/'events.jsonl').open('w') as record:
                for event in stream:
                    record.write(event.model_dump_json()+'\n');record.flush()
                    events+=1
                    if hasattr(event,'response'):
                        (round_dir/'last-response.json').write_text(event.response.model_dump_json(indent=2))
                    if event.type in ['response.completed','response.incomplete','response.failed']:
                        response=event.response
                    if events%100==0:
                        (round_dir/'stream-progress.json').write_text(json.dumps({'events':events,'elapsed_seconds':time.monotonic()-started}))
            if response is None:raise RuntimeError('No terminal response; all partial events retained')
    (round_dir / 'response.json').write_text(response.model_dump_json(indent=2))
    assert response.status == 'completed', response.status
    calls = [x for x in response.output if x.type == 'function_call']
    assert len(calls) == 1 and calls[0].name == 'submit_scene_candidate'
    result = json.loads(calls[0].arguments)
    (round_dir / 'candidate.json').write_text(json.dumps(result, indent=2))
    (round_dir / 'generated.py').write_text(result['program'])
    (round_dir / 'route_plan.json').write_text(json.dumps(result['route_plan'], indent=2))
    (round_dir / 'assessment.json').write_text(json.dumps(result['assessment'], indent=2))
    (round_dir / 'usage.json').write_text(json.dumps({'model': response.model, 'reasoning_effort': request['reasoning']['effort'],
        'usage': response.usage.model_dump(), 'wall_seconds': time.monotonic() - started, 'call_id': calls[0].call_id}, indent=2))
    compile(result['program'], str(round_dir / 'generated.py'), 'exec')
    print('GPT_SCENE_PROGRAM_RECEIVED', len(result['program']), flush=True)
except Exception as e:
    (round_dir / 'api-error.json').write_text(json.dumps({'type': type(e).__name__, 'status': getattr(e, 'status_code', None)}, indent=2))
    raise SystemExit('Asset authoring request failed; preserved sanitized record')
