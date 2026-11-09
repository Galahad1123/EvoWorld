"""Reload and render the saved official SAGE scene on the remote GPU."""
import os,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=Path(os.environ.get("EVOWORLD_RUN_DIR", str(root/"experiments/sage-bringup-20261107")))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"width":960,"height":720})
try:
    import omni.usd
    import omni.replicator.core as rep
    from pxr import UsdLux,UsdPhysics,UsdGeom,Gf
    import numpy as np
    from PIL import Image
    assert omni.usd.get_context().open_stage(str(out/"sage-scene-after.usda"))
    for _ in range(20): app.update()
    stage=omni.usd.get_context().get_stage()
    light=UsdLux.DomeLight.Define(stage,"/World/InspectionLight")
    light.CreateIntensityAttr(1000)
    ceiling=UsdLux.RectLight.Define(stage,"/World/InspectionCeilingLight")
    ceiling.CreateIntensityAttr(1000)
    ceiling.CreateWidthAttr(3)
    ceiling.CreateHeightAttr(5)
    UsdGeom.Xformable(ceiling).AddTranslateOp().Set(Gf.Vec3d(2.5,3.5,2.6))
    camera=rep.create.camera(position=(0.4,0.5,2.3),look_at=(3,4,1),focal_length=16)
    product=rep.create.render_product(camera,(960,720))
    rgb=rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(product)
    rep.orchestrator.step(rt_subframes=8,pause_timeline=True)
    data=rgb.get_data()
    assert data.shape[:2]==(720,960),data.shape
    assert float(data[:,:,:3].std())>10,"Render has no useful variation"
    Image.fromarray(data).save(out/"sage-scene-preview.png")
    report={"camera_position":[0.4,0.5,2.3],"camera_look_at":[3,4,1],"inspection_lighting_added":True,"image_shape":list(data.shape),"pixel_std":float(data[:,:,:3].std()),"rigid_body_count":sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()),"meters_per_unit":UsdGeom.GetStageMetersPerUnit(stage)}
    assert report["rigid_body_count"]==41,report
    (out/"sage-scene-render.json").write_text(json.dumps(report,indent=2))
    print("EVOWORLD_SAGE_RELOAD_RENDER_PASS",json.dumps(report),flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    (out/"sage-render-error.txt").write_text(traceback.format_exc())
    raise
finally:
    app.close()
