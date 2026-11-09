"""One bounded real GPT vision call; image estimates are not physical measurements."""
import base64
import json
from pathlib import Path
import sys
import time
from openai import OpenAI

out = Path(sys.argv[1]).resolve()
prompt = '''Analyze these two views from a SIMULATED SAGE scene, not real photographs.
Our scope is the blue rectangular table and the small red block only. Exclude robot, chairs, walls, and floor appearance.
The table is an existing textured asset that will be reused unchanged. Known asset bounds: length X=1.38532997 m,
width Y=0.79776659 m, top Z=0.79778135 m. These are source-asset dimensions, not real measurements.
Estimate the visible red block's size (XYZ in metres), base color RGB [0,1], and approximate XY position on the table.
Choose a NEW canonical scene frame: table centered at XY=(0,0), +X along its long side, +Y along short side.
These uncalibrated pictures cannot determine axis signs reliably: state the ambiguity, make a reasonable layout estimate,
and do not claim it reproduces a calibrated source pose. Geometry may be represented by a box. Preserve visible size/proportions.
Return ONLY a JSON object with keys cube_size_m (3 numbers), cube_xy_m (2 numbers), cube_color_rgb (3 numbers),
cube_yaw_degrees (number), visible_material (string), observations (string), uncertainties (array of strings).
Do not infer measured mass, friction or hidden structure. Physics properties will be separately labeled designed defaults.
This is a bounded geometric proxy workflow, not a claim of exact photorealistic reconstruction.'''
request = {'model': 'gpt-6-astra', 'reasoning': {'effort': 'low'}, 'max_output_tokens': 2048,
           'text': {'format': {'type': 'json_object'}}}
content = [{'type': 'input_text', 'text': prompt}]
for name in ['left.png', 'right.png']:
    image = out / 'input-views' / name
    content.append({'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode(), 'detail': 'high'})
(out / 'gpt-request.json').write_text(json.dumps({**request, 'prompt': prompt, 'image_files': ['input-views/left.png', 'input-views/right.png']}, indent=2))
started = time.monotonic()
try:
    with OpenAI(timeout=90, max_retries=0) as client:
        response = client.responses.create(**request, input=[{'role': 'user', 'content': content}])
    (out / 'gpt-response.json').write_text(response.model_dump_json(indent=2))
    if response.status != 'completed':
        raise RuntimeError('Incomplete GPT response; no scene will be built')
    estimate = json.loads(response.output_text)
    (out / 'image-estimate.json').write_text(json.dumps(estimate, indent=2))
    (out / 'gpt-usage.json').write_text(json.dumps({'model': response.model, 'reasoning_effort': 'low',
        'requests': 1, 'usage': response.usage.model_dump(), 'wall_seconds': time.monotonic() - started}, indent=2))
    print(json.dumps(estimate), flush=True)
except Exception as e:
    (out / 'gpt-error.json').write_text(json.dumps({'type': type(e).__name__, 'http_status': getattr(e, 'status_code', None)}, indent=2))
    raise SystemExit('GPT estimate failed; inspect preserved sanitized error record')
