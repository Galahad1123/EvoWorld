"""Target-engine photo-view inspection with stopped physics; no synthetic motion labels."""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

candidate=Path(sys.argv[1]).resolve()
out=candidate/'preview';out.mkdir(exist_ok=False)
app=None;status=1;start=time.monotonic()
try:
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'width':960,'height':960})
    import omni.usd
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, Gf
    from PIL import Image
    assert omni.usd.get_context().open_stage(str(candidate/'relocated/scene.usda'))
    stage=omni.usd.get_context().get_stage()
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Set(False)
    light=UsdLux.RectLight.Define(stage,'/World/InspectionLight');light.CreateIntensityAttr(1200.)
    light.CreateWidthAttr(1.5);light.CreateHeightAttr(1.5)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(-.2,-.25,1.5))
    cameras=json.loads((candidate/'cameras.json').read_text())
    eye=np.array([.45,-.60,.48]);target=np.array([-.03,.06,.045]);forward=target-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
    m=np.eye(4);m[:3,:3]=np.array([right,down,forward]).T;m[:3,3]=eye
    cameras['overview']={'world_T_camera_cv':m.tolist(),'image_size_wh':[960,720],'focal_length_px':900.}
    views={}
    for name,c in cameras.items():
        w,h=c['image_size_wh'];scale=min(960/w,960/h);res=(round(w*scale),round(h*scale))
        cam=UsdGeom.Camera.Define(stage,'/World/Inspect_'+name)
        mat=np.asarray(c['world_T_camera_cv'])@np.diag([1,-1,-1,1])
        UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(mat.T.tolist()))
        cam.CreateHorizontalApertureAttr(20.955);cam.CreateVerticalApertureAttr(20.955*h/w)
        cam.CreateFocalLengthAttr(float(c['focal_length_px']/w*20.955));cam.CreateClippingRangeAttr(Gf.Vec2f(.005,100.))
        product=rep.create.render_product(str(cam.GetPath()),res)
        ann=rep.AnnotatorRegistry.get_annotator('rgb');ann.attach(product)
        views[name]=(ann,res)
    for name,(ann,res) in views.items():
        rep.orchestrator.step(rt_subframes=4,delta_time=0.,pause_timeline=True,wait_for_render=True)
        pixels=ann.get_data()[:,:,:3].copy()
        assert pixels.shape==(res[1],res[0],3) and pixels.std()>5
        Image.fromarray(pixels).save(out/(name+'.png'))
    light.GetIntensityAttr().Set(250.)
    rep.orchestrator.step(rt_subframes=4,delta_time=0.,pause_timeline=True,wait_for_render=True)
    Image.fromarray(views['overview'][0].get_data()[:,:,:3]).save(out/'overview-low-light.png')
    (out/'report.json').write_text(json.dumps({'completed':True,'semantics':'static asset inspection; physics disabled, delta_time0',
        'cameras':cameras,'wall_seconds':time.monotonic()-start},indent=2))
    print('PHOTO_SCENE_RENDER_COMPLETE',flush=True);status=0
except BaseException:
    (out/'error.json').write_text(json.dumps({'traceback':traceback.format_exc()},indent=2));traceback.print_exc()
finally:
    if app is not None:app.close(exit_code=status)
    raise SystemExit(status)
