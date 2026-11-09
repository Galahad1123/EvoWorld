import asyncio
import json
import os
from pathlib import Path
import tempfile
from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client
from generate import decode_result
from test_scene_contract import valid_plan

def test_real_mcp_build_and_reject_premature_completion():
    asyncio.run(_exercise())

async def _exercise():
    root=Path(__file__).resolve().parents[3]
    with tempfile.TemporaryDirectory(prefix='test-mcp-',dir=root/'experiments') as td:
        run=Path(td);(run/'prompt.txt').write_text('Small meeting space test')
        env={k:os.environ[k] for k in ['HOME','PATH'] if k in os.environ};env['EVOWORLD_RUN_DIR']=td
        params=StdioServerParameters(command=str(root/'envs/sage/bin/python'),args=[str(root/'adapters/minimal_sage/server.py')],env=env,cwd=str(root))
        with (run/'server.log').open('w') as log:
          async with stdio_client(params,errlog=log) as (read,write):
           async with ClientSession(read,write) as client:
            await client.initialize()
            assert {x.name for x in (await client.list_tools()).tools}=={'list_assets','build_scene','simulate_scene','finish_scene'}
            catalog=decode_result(await client.call_tool('list_assets',{}))['assets']
            plan=valid_plan();kinds={a['kind']:a for a in catalog}
            for obj in plan['objects']:
                obj['asset_id']=kinds[obj['asset_id']]['asset_id']
            plan['objects'][3]['z']=kinds['table']['dimensions']['height']+.01
            built=decode_result(await client.call_tool('build_scene',{'plan':plan}))
            assert built['mesh_count']==10
            assert Path(built['layout_path']).is_file()
            result=await client.call_tool('finish_scene',{'scene_id':built['scene_id'],'visual_ok':True,'visual_notes':'Unverified claim must be rejected'})
            assert result.isError and not (run/'completion.json').exists()
            result=await client.call_tool('simulate_scene',{'scene_id':'../../outside'})
            assert result.isError

def test_model_image_input_preserves_initial_state_label(tmp_path):
    import base64
    from generate import make_image_input
    png=tmp_path/'preview.png'
    png.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='))
    message=make_image_input({'preview_path':str(png),'scene_id':'abc','rendered_state':'initial layout reloaded from saved USD; physics trajectories stored separately'},tmp_path)
    text=message['content'][0]['text']
    assert 'initial layout' in text and 'not a post-simulation image' in text
