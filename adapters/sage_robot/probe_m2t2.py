"""Real SAGE renderer -> official M2T2 checkpoint -> grasp poses; no substituted predictions."""
import hashlib,json,sys,tempfile,time,traceback
from pathlib import Path
import numpy as np
import torch
from evoworld.config import load,resource
from evoworld.expert import compiler_environment
import os
config=load();os.environ.update(compiler_environment(config))
root=Path(config['storage_root']);source=resource('expert_source')
run_root=Path(config['paths']['runs']);run_root.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(source/'server'))
out=Path(tempfile.mkdtemp(prefix='sage-original-m2t2-',dir=run_root));print(out,flush=True)
started=time.monotonic()
try:
 from pointnet2_ops import pointnet2_utils as ops
 torch.manual_seed(0);np.random.seed(0)
 xyz=torch.rand(1,128,3,device='cuda')
 ids=ops.furthest_point_sample(xyz,16);assert ids.shape==(1,16) and len(torch.unique(ids))==16
 feat=torch.randn(1,3,128,device='cuda',requires_grad=True)
 gathered=ops.gather_operation(feat,ids);gathered.sum().backward();torch.cuda.synchronize()
 assert torch.isfinite(gathered).all() and torch.isfinite(feat.grad).all()
 print('POINTNET_CUDA_PASS',flush=True)
 from tex_utils import dict_to_floor_plan,export_layout_to_mesh_dict_list_tree_search_with_object_id
 from m2t2_utils.data import generate_m2t2_data
 from m2t2_utils.infer import load_m2t2,infer_m2t2
 layout_path=source/'server/results/layout_evoworld_30dd907ee170/layout_evoworld_30dd907ee170.json'
 layout=dict_to_floor_plan(json.loads(layout_path.read_text()))
 target='room_evoworld_30dd907ee170_mug'
 meshes=export_layout_to_mesh_dict_list_tree_search_with_object_id(layout,target)
 meta,vis=generate_m2t2_data(meshes,target,[1.45,1.97,.8])
 vis['rgb'].save(out/'rgb.png');vis['seg'].save(out/'seg.png');np.save(out/'depth.npy',vis['depth'])
 model,cfg=load_m2t2()
 grasps,contacts=infer_m2t2(meta,vis,model,cfg,return_contacts=True)
 np.savez_compressed(out/'grasps.npz',grasps=grasps,contacts=contacts)
 assert len(grasps)>0 and grasps.shape[1:]==(4,4) and np.isfinite(grasps).all()
 result={'passed':True,'target':target,'grasps':len(grasps),'shape':list(grasps.shape),'wall_seconds':time.monotonic()-started,
         'checkpoint_sha256':hashlib.sha256(resource('m2t2').read_bytes()).hexdigest(),
         'model_runs':cfg.eval.num_runs,'torch':torch.__version__,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
 (out/'report.json').write_text(json.dumps(result,indent=2));print('SAGE_M2T2_PASS '+json.dumps(result),flush=True)
except BaseException:
 (out/'error.json').write_text(json.dumps({'traceback':traceback.format_exc(),'wall_seconds':time.monotonic()-started},indent=2));raise
