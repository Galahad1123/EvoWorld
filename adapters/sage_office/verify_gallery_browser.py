"""Exercise the offline gallery in an isolated Firefox WebDriver session."""
import argparse,base64,json,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('gallery',type=Path);p.add_argument('--session-file',required=True,type=Path);args=p.parse_args()
gallery=args.gallery.resolve();manifest=json.loads((gallery/'manifest.json').read_text())
session=json.loads(args.session_file.read_text())['value']['sessionId']
base='http://127.0.0.1:4447/session/'+session

def call(path,data=None,method=None):
 body=None if data is None else json.dumps(data).encode()
 req=urllib.request.Request(base+path,data=body,headers={'Content-Type':'application/json'},method=method)
 with urllib.request.urlopen(req,timeout=45) as r:result=json.load(r)
 value=result.get('value')
 if isinstance(value,dict) and value.get('error'):raise RuntimeError(value)
 return value

def js(script):return call('/execute/sync',{'script':script,'args':[]})
checks={}
try:
 call('/window/rect',{'width':1440,'height':1050})
 call('/url',{'url':(gallery/'index.html').as_uri()})
 n=js("return document.querySelectorAll('.card').length")
 assert n==manifest['scene_count'];checks['card_count']=n
 js("document.querySelectorAll('img').forEach(x=>x.loading='eager')")
 for _ in range(30):
  loaded=js("return [...document.querySelectorAll('.preview')].every(x=>x.complete&&x.naturalWidth>0)")
  if loaded:break
  time.sleep(.2)
 assert loaded;checks['first_view_images']=True
 # Exercise every alternate view, not just one representative button.
 js("document.querySelectorAll('.card').forEach(c=>c.querySelectorAll('.view')[1].click())")
 for _ in range(30):
  loaded=js("return [...document.querySelectorAll('.preview')].every(x=>x.complete&&x.naturalWidth>0)")
  if loaded:break
  time.sleep(.2)
 assert loaded
 assert js("return [...document.querySelectorAll('.card')].every(c=>c.querySelector('.preview-link').href===c.querySelector('.preview').src&&c.querySelectorAll('.view')[1].getAttribute('aria-pressed')==='true')")
 checks['second_views_and_fullsize_links']=True
 js("document.querySelectorAll('.card').forEach(c=>c.querySelectorAll('.view')[0].click());document.querySelector('#clear').click();document.querySelectorAll('.choice input')[0].click();document.querySelectorAll('.choice input')[8].click()")
 expected=[manifest['scenes'][i]['layout_id'] for i in [0,8]]
 assert all(x in js("return document.querySelector('#selection').value") for x in expected)
 call('/refresh',{})
 assert js("return document.querySelectorAll('.choice input:checked').length")==2
 checks['selection_and_reload_persistence']=True
 js("document.querySelector('#search').value='北欧';document.querySelector('#search').dispatchEvent(new Event('input'))")
 visible=js("return [...document.querySelectorAll('.card:not([hidden])')].map(x=>x.dataset.search)")
 assert visible and all('北欧' in x for x in visible);checks['chinese_style_search']=True
 js("document.querySelector('#search').value='no_matching_scene_xyz';document.querySelector('#search').dispatchEvent(new Event('input'))")
 assert js("return !document.querySelector('#empty').hidden")
 js("document.querySelector('#search').value='';document.querySelector('#search').dispatchEvent(new Event('input'));document.querySelector('#only-selected').click()")
 assert js("return document.querySelectorAll('.card:not([hidden])').length")==2
 checks['empty_state_and_selected_filter']=True
 # Intercept the Blob download to verify the exported payload, without a dialog.
 js("window.__downloadBlob=null;window.__clickedDownload=null;URL.createObjectURL=b=>{window.__downloadBlob=b;return 'blob:test';};HTMLAnchorElement.prototype.click=function(){window.__clickedDownload=this.download;};document.querySelector('#download').click()")
 assert js("return window.__clickedDownload")=='EvoWorld-office-selection.txt'
 exported=call('/execute/async',{'script':"const done=arguments[arguments.length-1];window.__downloadBlob.text().then(done);",'args':[]})
 assert all(x in exported for x in expected);checks['export_payload']=True
 # Force the clipboard-denied path that can occur with file:// pages.
 js("Object.defineProperty(navigator,'clipboard',{configurable:true,value:{writeText:()=>Promise.reject(new Error('test denied'))}});document.querySelector('#copy').click()")
 time.sleep(.1)
 assert 'Ctrl+C' in js("return document.querySelector('#status').textContent")
 assert js("return document.querySelector('#selection').selectionEnd>document.querySelector('#selection').selectionStart")
 checks['clipboard_manual_fallback']=True
 js("document.querySelector('#clear').click();document.querySelector('#only-selected').click()")
 assert js("return document.querySelectorAll('.choice input:checked').length")==0
 checks['clear_selection']=True
 call('/refresh',{})
 call('/window/rect',{'width':1440,'height':1050})
 call('/execute/async',{'script':"const done=arguments[arguments.length-1];Promise.all([...document.querySelectorAll('.preview')].slice(0,3).map(x=>x.decode())).then(()=>requestAnimationFrame(()=>requestAnimationFrame(()=>done(true))));",'args':[]})
 time.sleep(.3)
 (gallery/'browser-desktop.png').write_bytes(base64.b64decode(call('/screenshot')))
 call('/window/rect',{'width':390,'height':844})
 assert js("return document.documentElement.scrollWidth<=window.innerWidth")
 (gallery/'browser-mobile.png').write_bytes(base64.b64decode(call('/screenshot')))
 checks['mobile_no_horizontal_overflow']=True
 report={'passed':True,'browser':'isolated headless Firefox','transport':'local file://, no HTTP server','checks':checks}
 (gallery/'browser-verification.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report))
finally:
 call('',method='DELETE')
