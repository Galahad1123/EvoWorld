"""EvoWorld Isaac startup and SAGE extension probe (run on 5090-1)."""
import argparse, os, time, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--sage", action="store_true")
p.add_argument("--serve", action="store_true")
p.add_argument("--seconds", type=int, default=1800)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("SLURM_JOB_ID", "evoworld-sage")
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 960, "height": 720})
import omni.usd
omni.usd.get_context().new_stage()
if a.sage:
    manager = app._app.get_extension_manager()
    manager.add_path(os.environ.get("SAGE_EXT_ROOT", str(root / "upstream/sage/server/isaacsim")))
    manager.set_extension_enabled_immediate("isaac.sim.mcp_extension", True)
    assert manager.is_extension_enabled("isaac.sim.mcp_extension"), "SAGE extension did not enable"
    from isaac_sim_mcp_extension.extension import MCPExtension
    print("EVOWORLD_SAGE_EXTENSION_READY", flush=True)
else:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    import numpy as np
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    cube = world.scene.add(DynamicCuboid(prim_path="/World/TestCube", position=np.array([0.,0.,1.]), scale=np.array([.1,.1,.1]), mass=0.1))
    world.reset()
    start, _ = cube.get_world_pose()
    for _ in range(120):
        world.step(render=True)
    end, _ = cube.get_world_pose()
    assert .03 < float(end[2]) < .08, f"Cube failed to settle: {end}"
    report = {"start": start.tolist(), "end": end.tolist(), "steps": 120, "physics_pass": True}
    out=Path(os.environ.get("EVOWORLD_RUN_DIR", str(root/"experiments/sage-bringup-20261107")))
    (out/"isaac-physics.json").write_text(json.dumps(report, indent=2))
    omni.usd.get_context().get_stage().GetRootLayer().Export(str(out/"isaac-physics.usda"))
    print("EVOWORLD_ISAAC_PHYSICS_PASS", json.dumps(report), flush=True)
if a.serve:
    deadline=time.monotonic()+a.seconds
    print("EVOWORLD_SERVER_READY", flush=True)
    while app.is_running() and time.monotonic()<deadline:
        app.update()
else:
    for _ in range(20): app.update()
app.close()
