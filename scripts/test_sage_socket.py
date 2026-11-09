"""Validate real SAGE socket requests and their USD-stage side effects."""
import os, sys, json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/"upstream/sage/server/isaacsim"))
os.environ.setdefault("SLURM_JOB_ID","evoworld-sage")
from isaac_mcp.server import get_isaac_connection
out=Path(os.environ.get("EVOWORLD_RUN_DIR", str(root/"experiments/sage-bringup-20261107")))
marker=out/"mcp-side-effect.json"
if marker.exists(): marker.unlink()
conn=get_isaac_connection()
pong=conn.send_command("get_scene_info")
assert pong["status"]=="success", pong
code = f"""
import json, os
from pathlib import Path
from isaac_sim_mcp_extension.scene.utils import SERVER_ROOT_DIR
assert Path(SERVER_ROOT_DIR).resolve() == Path(os.environ["SAGE_SERVER_ROOT"]).resolve()
assert os.getpid() == {int(os.environ["EVOWORLD_EXPECTED_SERVER_PID"])!r}, "Wrong SAGE server PID"
assert os.environ.get("EVOWORLD_RUN_DIR") == {str(out)!r}, "Wrong SAGE server run directory"
stage=omni.usd.get_context().get_stage()
cube=UsdGeom.Cube.Define(stage,"/World/MCPProbe")
cube.CreateSizeAttr(0.1)
stage.GetRootLayer().Export({str(out/"mcp-probe.usda")!r})
Path({str(marker)!r}).write_text(json.dumps({{"prim":str(cube.GetPath()),"size":cube.GetSizeAttr().Get(),"server_root":SERVER_ROOT_DIR,"server_pid":os.getpid(),"run_dir":os.environ.get("EVOWORLD_RUN_DIR")}}))
"""
result=conn.send_command("execute_script",{"code":code})
assert result["status"]=="success", result
evidence=json.loads(marker.read_text())
assert evidence["prim"]=="/World/MCPProbe" and evidence["size"]==0.1
(out/"mcp-smoke.json").write_text(json.dumps({"pong":pong,"execute_script":result,"side_effect":evidence},indent=2))
conn.disconnect()
print("EVOWORLD_SAGE_SOCKET_PASS",json.dumps(evidence))
