"""Actual visual feedback call: model returns auditable edits to its generated asset program."""
import base64
import json
from pathlib import Path
import sys
import time
from openai import OpenAI

run=Path(sys.argv[1]).resolve();previous=run/sys.argv[2];out=run/sys.argv[3];out.mkdir(exist_ok=False)
prompt=Path(sys.argv[4]).read_text()
prompt+='\n\nEXECUTED SOURCE PROGRAM:\n'+(previous/'generated.py').read_text()
prompt+='\n\nCAMERA FIT RESULTS:\n'+(previous/'build/cameras.json').read_text()
content=[{'type':'input_text','text':prompt}]
paths=[run/'input'/n for n in ['main.jpg','left.jpg','right.jpg']]
paths += [previous/'build/preview'/n for n in ['main-comparison.png','left-comparison.png','right-comparison.png','overview.png']]
for p in paths:
    content.append({'type':'input_text','text':str(p.relative_to(run))})
    mime='image/jpeg' if p.suffix=='.jpg' else 'image/png'
    content.append({'type':'input_image','image_url':f'data:{mime};base64,'+base64.b64encode(p.read_bytes()).decode(),'detail':'high'})
tool={'type':'function','name':'review_and_patch_scene','description':'Review real target-engine photo comparisons and supply exact source replacements to execute in a new preserved candidate.',
    'parameters':{'type':'object','properties':{'verdict':{'type':'string','enum':['revise','accept_limited','reject']},'assessment':{'type':'object'},
        'patches':{'type':'array','items':{'type':'object','properties':{'old':{'type':'string'},'new':{'type':'string'},'reason':{'type':'string'}},'required':['old','new','reason'],'additionalProperties':False}}},'required':['verdict','assessment','patches'],'additionalProperties':False}}
request={'model':'gpt-6-astra','reasoning':{'effort':'medium'},'max_output_tokens':6000,'tools':[tool],'tool_choice':{'type':'function','name':'review_and_patch_scene'}}
(out/'request.json').write_text(json.dumps({**request,'prompt':prompt,'images':[str(p.relative_to(run)) for p in paths]},indent=2))
start=time.monotonic();response=None
try:
    with OpenAI(timeout=600,max_retries=0) as client:
        with client.responses.create(**request,input=[{'role':'user','content':content}],stream=True) as stream:
            with (out/'events.jsonl').open('w') as events:
                for event in stream:
                    events.write(event.model_dump_json()+'\n');events.flush()
                    if event.type in ['response.completed','response.incomplete','response.failed']:response=event.response
    assert response is not None,'No terminal response'
    (out/'response.json').write_text(response.model_dump_json(indent=2))
    assert response.status=='completed',response.status
    call=next(x for x in response.output if x.type=='function_call' and x.name=='review_and_patch_scene')
    result=json.loads(call.arguments);(out/'review.json').write_text(json.dumps(result,indent=2))
    (out/'usage.json').write_text(json.dumps({'model':response.model,'reasoning_effort':'medium','usage':response.usage.model_dump(),'wall_seconds':time.monotonic()-start,'call_id':call.call_id},indent=2))
    program=(previous/'generated.py').read_text()
    for patch in result['patches']:
        assert program.count(patch['old'])==1,('Patch must match exactly once',patch['reason'])
        program=program.replace(patch['old'],patch['new'],1)
    compile(program,str(out/'generated.py'),'exec')
    (out/'generated.py').write_text(program)
    (out/'route_plan.json').write_text((previous/'route_plan.json').read_text())
    print('GPT_VISUAL_REVIEW_RECEIVED',result['verdict'],'patches',len(result['patches']),flush=True)
except Exception as e:
    (out/'api-error.json').write_text(json.dumps({'type':type(e).__name__,'status':getattr(e,'status_code',None)},indent=2))
    raise SystemExit('Visual review failed; partial response retained')
