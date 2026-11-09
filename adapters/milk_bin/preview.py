"""Preview the approved office with the new box; exercise its physical cavity."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'adapters/sage_office/preview.py'
text=source.read_text().replace('import photo_runtime as photo','import milk_runtime as photo').replace('import runtime as office','from milk_runtime import office')
anchor=' from pxr import Usd,UsdGeom,UsdPhysics\n'
assert text.count(anchor)==1
text=text.replace(anchor,''' from task_success import check_window
 from collision_bridge import collect_colliders
 from pxr import Usd
 milk_spec=next(o for o in photo.SPEC['objects'] if o['id']=='milk')
 vertices,_,paths=collect_colliders(Usd.Stage.Open(str(photo.BUNDLE/milk_spec['asset'])),'/Asset',True)
 box_vertices,_,box_paths=collect_colliders(Usd.Stage.Open(str(photo.BOX_ASSET)),'/Asset',True)
 assert len(box_paths)==5
 drop_checks={}
 for kind,offset in [('inside',[0,0,.08]),('outside',[.13,0,.08])]:
  state=torch.zeros((1,13),device=env.device);state[0,:3]=tensor(photo.BOX_POS)+tensor(offset);state[0,3]=1
  env.scene['milk'].write_root_state_to_sim(state);env.scene['milk'].reset()
  roots=[];fingers=[]
  for _ in range(150):
   env.step(hold);roots.append(photo.old.cpu(env.scene['milk'].data.root_state_w[0]).copy());fingers.append(photo.old.cpu(robot.data.joint_pos[0,-2:]).copy())
  result=check_window(roots,fingers,vertices,photo.BOX_POS,photo.BOX_INNER,photo.FLOOR,env.step_dt,gripper_commands=np.ones(len(roots)))
  drop_checks[kind]=result
  assert result['success']==(kind=='inside'),(kind,result)
  camera=env.scene['camera_tabletop_preview'];camera.update(0.,force_recompute=True)
  Image.fromarray(photo.old.cpu(camera.data.output['rgb'][0])).save(out/(kind+'-drop.png'))
 (out/'box-physical-checks.json').write_text(json.dumps({'collision_parts':box_paths,'checks':drop_checks,'step_dt':env.step_dt},indent=2))
'''+anchor)
text=text.replace("for name in ['preview.py','runtime.py']:","for name in ['preview.py','milk_runtime.py']:")
exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':__file__})
