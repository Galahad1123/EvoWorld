"""Official RLinf conversion kernels; task statistics added after train data exists."""
import json,gc,hashlib
from pathlib import Path
from rlinf.utils.ckpt_convertor.openpi import jax_to_openpi as conversion
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'models/pi05-base-jax';out=ROOT/'models/pi05-base-openpi'
assert json.loads((source/'download-verified.json').read_text())['all_gcs_crc32c_verified']
assert not (out/'model.safetensors').exists()
params=conversion._load_jax_params(source)
merged={}
for fn in [conversion.convert_siglip,lambda p:conversion.convert_llm(p,True),lambda p:conversion.convert_projections(p,True)]:
 for name,value in fn(params).items():merged[name]=value.contiguous()
del params;gc.collect()
conversion.save_safetensors(merged,out/'model.safetensors')
conversion.write_config_json({'action_dim':32,'action_horizon':20,'max_token_len':200,'paligemma_variant':'gemma_2b','action_expert_variant':'gemma_300m','pi05':True,'pcd':False,'dtype':'bfloat16'},out)
(out/'conversion-provenance.json').write_text(json.dumps({'source':'gs://openpi-assets/checkpoints/pi05_base','conversion_module':str(Path(conversion.__file__)),'conversion_module_sha256':hashlib.sha256(Path(conversion.__file__).read_bytes()).hexdigest(),'tensors':len(merged),'parameters':sum(v.numel() for v in merged.values()),'norm_stats_status':'not present; must compute exclusively from new training demonstrations'},indent=2))
print('CONVERSION_COMPLETE',flush=True)
