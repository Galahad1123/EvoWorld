"""Join physically accepted photo demonstrations and set authoritative scene metadata."""
import hashlib,json,subprocess,sys
from pathlib import Path
import h5py
root=Path(__file__).resolve().parents[2]
run=Path(sys.argv[1]).resolve();pilot=Path(sys.argv[2]).resolve()
(run/'prepare-photo-executed.py').write_bytes(Path(__file__).read_bytes())
for source,count in [(run,18),(pilot,2)]:
 subprocess.run([sys.executable,str(root/'adapters/sage_photo_dp/acceptance.py'),str(source)],check=True)
 report=json.loads((source/'photo-physical-verification.json').read_text())
 assert report['all_pass'] and report['total']==count
subprocess.run([sys.executable,str(root/'adapters/sage_dp/prepare.py'),str(run),'--extra-run',str(pilot)],check=True)
raw=next(run.glob('robomimic_data/*/experts.hdf5'))
with h5py.File(raw,'r') as f:attrs=dict(f['data'].attrs)
with h5py.File(run/'training.hdf5','r+') as f:
 # prepare.py copies attrs from the final source; use current configured scene
 # instead of the early pilot's stale pre-override env.yaml metadata.
 for key,value in attrs.items():
  if key!='total':f['data'].attrs[key]=value
 f['data'].attrs['evoworld_scene_profile']=(run/'scene-profile.json').read_text()
 for name in f['data']:
  g=f['data'][name]
  for key in ['rgb_tabletop','rgb_wrist']:
   assert g['obs/'+key].shape[1:]==(128,128,3)
report=json.loads((run/'dataset-report.json').read_text())
h=hashlib.sha256()
with (run/'training.hdf5').open('rb') as stream:
 for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
report['training_sha256']=h.hexdigest()
report['scene_metadata_source']=str(raw)
report['pilot_metadata_caveat']='Pilot saved pre-override cfg is stale; scene-profile/initial snapshots and executed source preserve actual scene. Derived HDF5 uses current configured metadata.'
report['pilot_m2t2_caveat']='Pilot M2T2 rasterizer atlas had constant-material tile bleeding; bulk fixes texel-center mapping. Native Isaac policy RGB was unaffected in both. Pilot provenance retained and physically audited.'
report['expert_physical_acceptance']='all20 passed exact saved interval checks, including angular velocity'
(run/'dataset-report.json').write_text(json.dumps(report,indent=2)+'\n')
print('PHOTO_DATASET_READY20',flush=True)
