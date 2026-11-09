"""Use SAGE's own per-object USDZ exporter to prepare the existing scene for IsaacLab."""
import json,os,sys,traceback
from pathlib import Path
root=Path(__file__).resolve().parents[2]
server=root/'adapters/sage_robot/src/server';os.environ['SAGE_SERVER_ROOT']=str(server)
layout_id=os.environ.get('EVOWORLD_LAYOUT_ID','layout_evoworld_30dd907ee170')
folder=server/'results'/layout_id;dest=folder/(layout_id+'_usd_collection')
if dest.exists():
 raise FileExistsError(f'Refusing to overwrite existing exported fixture: {dest}')
report_path=folder/'export-report.json'
report_path.write_text(json.dumps({'passed':False,'status':'started'}))
sys.path.insert(0,str(root/'adapters/isaac6/isaac.sim.mcp_extension'))
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
try:
 from isaac_sim_mcp_extension.extension import MCPExtension
 exporter=MCPExtension()
 result=exporter.get_room_layout_scene_usd_separate_from_layout(str(folder/(layout_id+'.json')),str(dest))
 assert result['status']=='success',result
 assert len(list(dest.glob('*.usdz')))==10
 masses=json.loads((dest/'rigid_object_property_dict.json').read_text());assert len(masses)==4
 (folder/'export-report.json').write_text(json.dumps({'passed':True,'objects':len(masses),'usdz_files':len(list(dest.glob('*.usdz')))},indent=2))
 print('SAGE_ORIGINAL_USDZ_PASS',flush=True)
except BaseException:
 traceback.print_exc()
 report_path.write_text(json.dumps({'passed':False,'traceback':traceback.format_exc()},indent=2))
 raise
finally:app.close()
