"""Run a bounded GPT Responses → actual MCP → SAGE → Isaac generation loop."""
import argparse
import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
import time
from datetime import datetime,timedelta
from pathlib import Path
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from acceptance import MCP_TIMEOUT_SECONDS,read_run_config,validate_run_config

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_PROMPT='Create a new small single-room meeting workspace, with one table near the center, two chairs on opposite sides, and one mug safely supported near the center of the tabletop. Choose a sensible room size and new object positions. Keep furniture separated, inside the room, and stable under gravity. Use only the supplied official assets.'
INSTRUCTIONS='''You are the EvoWorld scene-generation agent. Use only gpt-6-astra reasoning and the supplied tools.
First list_assets, then build_scene using its schema. You must generate a NEW layout, not reproduce old positions.
Then simulate_scene to run actual physics and independent USD reload/render. Read diagnostics; if failed, build a revised scene.
After simulate_scene, the real rendered image will be attached as a user input. Inspect the image for the requested furniture, tabletop mug, plausible placement and no obvious floating/intersection. Only then call finish_scene with honest visual_notes.
Never claim success based only on a tool exit code. Never fabricate an artifact or change mesh assets. At most 10 turns and 3 simulation attempts are available. A room with no doors/windows is intentional for this bounded test. Do not add extra objects.'''


def decode_result(result):
    texts=[c.text for c in result.content if c.type=='text']
    if result.isError:
        return {'error':True,'message':'\n'.join(texts)}
    if len(texts)==1:
        try: return json.loads(texts[0])
        except json.JSONDecodeError: pass
    return {'text':'\n'.join(texts)}

def make_image_input(result,run_dir):
    preview=Path(result['preview_path']).resolve()
    if not preview.is_relative_to(run_dir) or not preview.is_file():
        raise RuntimeError('Invalid preview artifact')
    state=result['rendered_state']
    image=base64.b64encode(preview.read_bytes()).decode()
    return {'role':'user','content':[
        {'type':'input_text','text':f'Actual Isaac rendering of scene {result["scene_id"]}. Rendered state: {state}. This is the initial layout, not a post-simulation image. Review its appearance separately from the measured physics trajectories in the preceding tool output.'},
        {'type':'input_image','image_url':'data:image/png;base64,'+image,'detail':'high'}]}

async def run(prompt,run_dir):
    config=read_run_config(run_dir)
    script_dir=Path(__file__).resolve().parent
    env={k:os.environ[k] for k in ['HOME','PATH','LANG','LC_ALL','OMP_NUM_THREADS'] if k in os.environ}
    env.update(EVOWORLD_RUN_DIR=str(run_dir),OMNI_KIT_ACCEPT_EULA='YES')
    params=StdioServerParameters(command=str(ROOT/'envs/sage/bin/python'),args=['-u',str(script_dir/'server.py')],env=env,cwd=str(ROOT))
    input_items=[{'role':'user','content':prompt}]
    previous=None;usage=[];images_sent=set()
    with (run_dir/'mcp-server.log').open('w') as server_log:
      async with stdio_client(params,errlog=server_log) as (read,write):
       async with ClientSession(read,write,read_timeout_seconds=timedelta(seconds=MCP_TIMEOUT_SECONDS)) as session:
        await session.initialize()
        listed=await session.list_tools()
        tools=[{'type':'function','name':t.name,'description':t.description or '', 'parameters':t.inputSchema,'strict':False} for t in listed.tools]
        (run_dir/'tools.json').write_text(json.dumps(tools,indent=2))
        async with AsyncOpenAI(timeout=180,max_retries=0) as client:
         for turn in range(1,config['max_turns']+1):
            print(f'GPT_TURN {turn}',flush=True)
            request={'model':'gpt-6-astra','reasoning':{'effort':'low'},'instructions':INSTRUCTIONS.replace('At most 10 turns and 3 simulation attempts are available.',f'At most {config["max_turns"]} turns and {config["max_simulations"]} simulation attempts are available. Do not regenerate after a failed physics attempt if the attempt budget is one.'),'tools':tools,
                'parallel_tool_calls':False,'max_output_tokens':2048,'input':input_items}
            if previous: request['previous_response_id']=previous
            (run_dir/f'request-{turn:02d}.json').write_text(json.dumps(request,indent=2))
            response=await client.responses.create(**request)
            (run_dir/f'response-{turn:02d}.json').write_text(response.model_dump_json(indent=2))
            if response.status!='completed':
                raise RuntimeError('GPT response was incomplete; partial function arguments will not be executed')
            usage.append(response.usage.model_dump())
            previous=response.id
            calls=[item for item in response.output if item.type=='function_call']
            if len(calls)!=1:
                raise RuntimeError('Expected exactly one actual tool call per turn before completion')
            call=calls[0]
            if call.name not in {t['name'] for t in tools}:
                raise RuntimeError('Unknown requested tool')
            arguments=json.loads(call.arguments)
            print(f'MCP_TOOL {call.name}',flush=True)
            result=decode_result(await session.call_tool(call.name,arguments))
            (run_dir/f'tool-{turn:02d}.json').write_text(json.dumps({'name':call.name,'arguments':arguments,'result':result},indent=2))
            if call.name=='simulate_scene' and config['max_simulations']==1 and not result.get('passed'):
                raise RuntimeError('One-attempt physics budget consumed; failure retained without regeneration')
            input_items=[{'type':'function_call_output','call_id':call.call_id,'output':json.dumps(result)}]
            if call.name=='simulate_scene' and not result.get('error') and result.get('preview_path'):
                input_items.append(make_image_input(result,run_dir))
                images_sent.add(result['scene_id'])
            if call.name=='finish_scene' and result.get('passed'):
                if result['scene_id'] not in images_sent:
                    raise RuntimeError('Completion lacks an actual image-review round')
                completion=json.loads((run_dir/'completion.json').read_text())
                if completion['scene_id']!=result['scene_id'] or not completion['passed']:
                    raise RuntimeError('Completion artifact mismatch')
                summary={'run_config':config,'passed':True,'model':response.model,'model_turns':turn,'usage':usage,'completion':completion,
                    'total_input_tokens':sum(u['input_tokens'] for u in usage),'total_output_tokens':sum(u['output_tokens'] for u in usage)}
                (run_dir/'summary.json').write_text(json.dumps(summary,indent=2))
                print('EVOWORLD_MINIMAL_GENERATION_PASS',json.dumps({'run_dir':str(run_dir),'scene_id':result['scene_id'],'model_turns':turn}),flush=True)
                return
    raise RuntimeError('Model turn limit reached without accepted completion')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prompt',default=DEFAULT_PROMPT)
    parser.add_argument('--physics-steps',type=int,default=120)
    parser.add_argument('--max-turns',type=int,default=10)
    parser.add_argument('--max-simulations',type=int,default=3)
    parser.add_argument('--run-parent',type=Path,default=ROOT/'experiments')
    args=parser.parse_args()
    config=validate_run_config({'physics_steps':args.physics_steps,'max_turns':args.max_turns,'max_simulations':args.max_simulations})
    parent=args.run_parent.resolve()
    if not parent.is_relative_to(ROOT/'experiments'): parser.error('Run parent must be within EvoWorld experiments')
    parent.mkdir(parents=True,exist_ok=True)
    run_dir=Path(tempfile.mkdtemp(prefix='sage-generated-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-',dir=parent))
    (run_dir/'prompt.txt').write_text(args.prompt)
    (run_dir/'run-config.json').write_text(json.dumps(config,indent=2))
    files=list(Path(__file__).resolve().parent.glob('*.py'))+list((ROOT/'assets/SAGE-10k/kits').glob('*.py'))
    manifest={'command':'scripts/with_gpt_env.py -- adapters/minimal_sage/generate.py','model':'gpt-6-astra',
        'sage_commit':subprocess.check_output(['git','-C',str(ROOT/'upstream/sage'),'rev-parse','HEAD'],text=True).strip(),
        'new_layout':True,'new_mesh_generation':False,'run_config':config,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('RUN_DIR='+str(run_dir),flush=True)
    started=time.monotonic()
    try:
        asyncio.run(run(args.prompt,run_dir))
    except Exception as exc:
        error={'error_type':type(exc).__name__,'http_status':getattr(exc,'status_code',None),'stack':traceback.format_tb(exc.__traceback__)}
        (run_dir/'error.json').write_text(json.dumps(error,indent=2));print(json.dumps({'status':'failed','run_dir':str(run_dir),'error_type':type(exc).__name__}),flush=True)
        sys.exit(1)

    finally:
        (run_dir/'timing.json').write_text(json.dumps({'wall_seconds':time.monotonic()-started},indent=2))
