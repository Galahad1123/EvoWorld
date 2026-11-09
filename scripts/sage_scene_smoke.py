"""Run the official SAGE-10k scene importer and stability test without an LLM."""
import os,sys,json,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"assets/SAGE-10k/kits"))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
import isaacsim_utils as official
import numpy as np
out=Path(os.environ.get("EVOWORLD_RUN_DIR", str(root/"experiments/sage-bringup-20261107")))
out.mkdir(exist_ok=True,parents=True)
official.start_simulation_app()
try:
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    scene=root/"assets/SAGE-10k/extracted/layout_84b703fb.json"
    track_ids=official.get_layout_scene_loaded(str(scene))
    print("TRACK_COUNT",len(track_ids),flush=True)
    assert len(track_ids)==59, len(track_ids)
    stage=omni.usd.get_context().get_stage()
    initial_rigid_count=sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse())
    print("RIGID_COUNT",initial_rigid_count,flush=True)
    assert initial_rigid_count > 0, initial_rigid_count
    assert all(stage.GetPrimAtPath("/World/"+name).IsValid() for name in track_ids)
    # Preserve the actual stage, including assets; do not silently correct units or physics.
    meters=UsdGeom.GetStageMetersPerUnit(stage)
    assert meters==1.0, meters
    stage.GetRootLayer().Export(str(out/"sage-scene-initial.usda"))
    result=official.simulate_the_scene(track_ids)
    stats=official.generate_physics_statistics(result["traced_data_all"],track_ids)
    def encode(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        raise TypeError(type(v).__name__)
    fixture_files=[scene,root/"assets/SAGE-10k/scenes/20251213_020526_layout_84b703fb.zip",*sorted((root/"assets/SAGE-10k/kits").glob("*.py"))]
    provenance={"dataset_revision":"d425de49ff445fd3903d790a11461d23ffd0c7dd","sha256":{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture_files}}
    report={"provenance":provenance,"source":"official SAGE-10k scene (not newly generated)","track_count":len(track_ids),"rigid_body_count":initial_rigid_count,"meters_per_unit":meters,"stats":stats,"result":result}
    (out/"sage-scene-physics.json").write_text(json.dumps(report,default=encode,indent=2))
    stage.GetRootLayer().Export(str(out/"sage-scene-after.usda"))
    assert result["status"]=="success", result
    assert len(result["traced_data_all"])==59
    reloaded=Usd.Stage.Open(str(out/"sage-scene-after.usda"))
    assert reloaded and sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in reloaded.Traverse())==initial_rigid_count
    print("EVOWORLD_SAGE_SCENE_TEST_EXECUTED",json.dumps({k:v for k,v in stats.items() if k!="objects"},default=encode),flush=True)
    print("METERS_PER_UNIT",meters,flush=True)
except BaseException as exc:
    import traceback
    traceback.print_exc()
    (out/"sage-scene-error.json").write_text(json.dumps({"error":repr(exc),"traceback":traceback.format_exc()},indent=2))
    raise
finally:
    official.simulation_app.close()
