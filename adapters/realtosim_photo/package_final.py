from pathlib import Path
import zipfile,json,hashlib,tempfile
from pxr import UsdUtils
import sys
r=Path(sys.argv[1]).expanduser().resolve();z=r/'scene-with-provenance.zip';files={}
for p in (r/'round3/build/relocated').rglob('*'):
 if p.is_file():files['scene/'+str(p.relative_to(r/'round3/build/relocated'))]=p
for sub in ['input','round3/build/physics','round3/build/preview']:
 for p in (r/sub).rglob('*'):
  if p.is_file():files[str(p.relative_to(r))]=p
for sub in ['round1-complete','round2','round3']:
 for name in ['generated.py','request.json','response.json','usage.json','review.json','route_plan.json','tool-execution.json']:
  p=r/sub/name
  if p.exists():files[str(p.relative_to(r))]=p
for name in ['README.md','acceptance.json','adapter-tests.log','upstream-tests.log','joint-fit-proposal.json','joint-fit-reproduced.json']:
 files[name]=r/name
for name in ['metadata.json','scene.json','observations.json','cameras.json','camera-selection.json','camera-fit-final.json','asset-check.json','physics-readback.json']:
 p=r/'round3/build'/name;files[str(p.relative_to(r))]=p
manifest={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
 for name,p in files.items():f.write(p,name)
 f.writestr('manifest-sha256.json',json.dumps(manifest,indent=2))
with tempfile.TemporaryDirectory(prefix='photo-final-package-') as tmp:
 with zipfile.ZipFile(z) as f:f.extractall(tmp)
 base=Path(tmp);layers,assets,missing=UsdUtils.ComputeAllDependencies(str(base/'scene/scene.usda'))
 assert not missing
 assert all(Path(p.realPath).resolve().is_relative_to(base) for p in layers)
 assert all(Path(str(p)).resolve().is_relative_to(base) for p in assets)
 for name,h in manifest.items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==h
report={'passed':True,'files':len(files),'bytes':z.stat().st_size,'sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'dependency_relocation_verified':True}
(r/'archive-check.json').write_text(json.dumps(report,indent=2));print(report)
