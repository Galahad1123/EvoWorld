"""Execute a reviewed GPT construction program, then original realtosim tools."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

run=Path(sys.argv[1]).resolve();round_dir=run/sys.argv[2];candidate=round_dir/'build'
candidate.mkdir(exist_ok=False)
script=round_dir/'generated.py'
env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['API_KEY','TOKEN','PASSWORD','SECRET'])}
env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
commands=[['generated-program',[sys.executable,str(script),str(candidate),str(run/'input')]],
          ['build-scene',[sys.executable,'-m','g1_realtosim','build-scene',str(candidate/'scene.json'),str(candidate/'assembled')]],
          ['package-assets',[sys.executable,'-m','g1_realtosim','package-assets',str(candidate/'assembled/scene.usda'),str(candidate/'packaged')]]]
records=[]
for name,cmd in commands:
    start=time.monotonic()
    with (round_dir/(name+'.log')).open('w') as f:
        p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=90)
    records.append({'name':name,'command':cmd,'exit_code':p.returncode,'seconds':time.monotonic()-start})
    (round_dir/'tool-execution.json').write_text(json.dumps({'program_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'executions':records},indent=2))
    if p.returncode:raise SystemExit(f'{name} failed; retained log')
    if name=='generated-program':
        # Original CLI requires declarative fields even when USD already owns the materials/physics.
        # Copy the model's metadata into those fields; never substitute geometry or physics values.
        spec_path=candidate/'scene.json';spec=json.loads(spec_path.read_text())
        metadata=json.loads((candidate/'metadata.json').read_text())
        shutil.copy2(spec_path,candidate/'scene.generated.json')
        for obj in spec['objects']:
            record=metadata['objects'][obj['id']]
            obj.setdefault('material',record['provenance']['appearance'])
            obj.setdefault('texture',record['provenance']['texture'])
            obj.setdefault('physics',{**record['physics'],'deformable':False,'rigid':record['dynamic']})
        spec_path.write_text(json.dumps(spec,indent=2))
shutil.copytree(candidate/'packaged',candidate/'relocated')
print('GPT_PROGRAM_AND_ORIGINAL_REALTOSIM_TOOLS_COMPLETE',candidate)
