"""Build a portable offline HTML gallery from official cached office previews."""
import argparse,csv,hashlib,html,json,re,shutil,zipfile
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
def esc(value):return html.escape(str(value),quote=True)

def style_label(style):
 for keyword,label in [('moroccan','摩洛哥风格'),('cyberpunk','赛博朋克风格'),('glam','华丽装饰风格'),('shabby chic','复古田园风格'),('scandinavian','北欧风格'),('art deco','艺术装饰风格'),('1950','50年代复古'),('vintage','复古风格'),('rustic','乡村木质风格'),('contemporary','当代风格'),('bohemian','波西米亚风格'),('neoclassical','新古典风格'),('futuristic','未来科技风格'),('wabi','侘寂风格'),('minimal','极简风格'),('industrial','工业风格'),('mid-century','中古风格'),('mediterranean','地中海风格'),('bauhaus','包豪斯风格'),('gothic','哥特风格'),('traditional','传统风格'),('modern','现代风格')]:
  if keyword in style.lower():return label
 return '办公场景'


def main():
 p=argparse.ArgumentParser();p.add_argument('collection',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 collection=json.loads((a.collection/'collection.json').read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 assert 10<=len(collection['found'])<=20
 scenes=[];cards=[]
 excluded=[r for r in collection['found'] if any(word in ' '.join(r['room_types']).lower() for word in ['dentist','dental','post office'])]
 candidates=[r for r in collection['found'] if r not in excluded]
 assert 10<=len(candidates)<=20
 for i,item in enumerate(candidates,1):
  scene_id=item['layout_id'];assert re.fullmatch(r'layout_[a-f0-9]+',scene_id)
  previews=item['previews'];source=Path(previews[0]['file']).parent
  layout=json.loads((source/(scene_id+'.json')).read_text());office_rooms=[r for r in layout['rooms'] if 'office' in r['room_type'].lower()];assert office_rooms
  style=layout.get('building_style','未注明风格');room=office_rooms[0];d=room['dimensions']
  dest=out/'images'/scene_id;dest.mkdir(parents=True)
  images=[]
  for k,entry in enumerate(previews[:2]):
   original=Path(entry['file']);assert hashlib.sha256(original.read_bytes()).hexdigest()==entry['sha256']
   with Image.open(original) as im:im.verify()
   image=dest/('view-'+str(k)+original.suffix);shutil.copyfile(original,image)
   images.append(image.relative_to(out).as_posix())
  assert len(images)==2
  details=out/'metadata';details.mkdir(exist_ok=True)
  (details/(scene_id+'.json')).write_text(json.dumps(layout,ensure_ascii=False,indent=2))
  row={'number':f'{i:02}','layout_id':scene_id,'style':style,'style_label_zh':style_label(style),'room_type':room['room_type'],'width_m':d['width'],'length_m':d['length'],'height_m':d['height'],'room_count':len(layout['rooms']),'object_count':sum(len(r['objects']) for r in layout['rooms']),'archive_path':item['path'],'archive_sha256':item.get('archive_sha256'),'dataset_revision':collection['dataset_revision'],'source_url':f"https://huggingface.co/datasets/nvidia/SAGE-10k/blob/{collection['dataset_revision']}/{item['path']}",'images':images,'preview_sha256':[p['sha256'] for p in previews[:2]],'already_composed':scene_id=='layout_a5ac58b7'}
  scenes.append(row)
  search=esc(' '.join([row['number'],scene_id,style,style_label(style),room['room_type']]))
  dimensions=f"{d['width']:g} × {d['length']:g} m"
  current='<span class="badge current">当前试装背景</span>' if row['already_composed'] else ''
  cards.append(f'''<article class="card" data-id="{scene_id}" data-number="{i:02}" data-search="{search}">
<div class="card-top"><label class="choice"><input type="checkbox" value="{scene_id}" aria-label="选择候选 {i:02}"><span>候选 {i:02}</span></label>{current}</div>
<a class="preview-link" href="{images[0]}" target="_blank" rel="noopener"><img class="preview" src="{images[0]}" alt="候选 {i:02} 官方预览，视角 1" loading="lazy" width="1920" height="1080"></a>
<div class="view-row"><div class="view-switch" aria-label="切换视角"><button type="button" class="view active" data-src="{images[0]}" data-view="1" aria-pressed="true">视角 1</button><button type="button" class="view" data-src="{images[1]}" data-view="2" aria-pressed="false">视角 2</button></div><span class="hint">点击图片看原图 ↗</span></div>
<div class="card-body"><h2>{esc(style_label(style))}</h2><p class="original-style">{esc(style)}</p><div class="specs"><span>{dimensions}</span><span>高 {d['height']:g} m</span><span>{row['object_count']} 件物品</span></div><code>{scene_id}</code>
<details><summary>查看场景说明与来源</summary><p>{esc(layout.get('description',''))}</p><p>房间类型：{esc(row['room_type'])} · 场景共 {row['room_count']} 个房间</p><a href="metadata/{scene_id}.json" target="_blank">布局信息 JSON</a> · <a href="{esc(row['source_url'])}" target="_blank" rel="noopener">官方场景包</a></details></div></article>''')
 count=len(scenes)
 template=r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EvoWorld · Office 场景候选</title>
<style>
:root{--ink:#172c3b;--muted:#607382;--line:#dce4e8;--accent:#196d78;--paper:#f2f5f5}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;font-size:15px;line-height:1.65}header,main,footer{max-width:1500px;margin:auto;padding:32px 40px}header{padding-top:48px;padding-bottom:22px}.eyebrow{color:var(--accent);letter-spacing:.16em;font-size:12px;font-weight:750}h1{font-size:clamp(28px,3.2vw,46px);line-height:1.25;letter-spacing:-.035em;margin:12px 0 16px}header p{color:var(--muted);max-width:890px;margin:6px 0}.count{display:inline-block;padding:3px 12px;background:#e2eef0;border-radius:20px;font-size:13px;color:#23535b;vertical-align:middle;margin-left:8px}a{color:var(--accent)}button,input,textarea{font:inherit}button{cursor:pointer;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink);padding:9px 14px}button:hover{border-color:var(--accent);background:#eef7f7}button:focus-visible,a:focus-visible,input:focus-visible,summary:focus-visible,textarea:focus-visible{outline:3px solid #e8a747;outline-offset:3px}.toolbar{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:22px}.search{flex:1;min-width:220px;padding:11px 14px;border:1px solid #cbd6dc;border-radius:7px;background:#fbfcfc}.only-selected{display:flex;gap:8px;align-items:center;font-size:14px}input[type=checkbox]{width:18px;height:18px;accent-color:var(--accent);cursor:pointer}.results-count{color:var(--muted);font-size:13px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}.card{background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 4px 18px #1e344308;transition:border-color .15s,box-shadow .15s}.card.selected{border:2px solid var(--accent);box-shadow:0 0 0 3px #196d7815}.card[hidden]{display:none}.card-top{padding:13px 15px;display:flex;align-items:center;justify-content:space-between;gap:8px}.choice{display:flex;align-items:center;gap:9px;font-weight:700;cursor:pointer}.badge{font-size:11px;padding:3px 8px;border-radius:20px;background:#e7f1ed;color:#29614e}.preview-link{display:block;background:#141a1f}.preview{display:block;width:100%;height:auto;aspect-ratio:16/9;object-fit:contain}.view-row{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid #edf1f3;gap:8px}.view-switch{display:flex;gap:5px}.view{font-size:12px;padding:5px 10px}.view.active{background:var(--ink);border-color:var(--ink);color:white}.hint{font-size:11px;color:var(--muted)}.card-body{padding:16px}.card h2{font-size:15px;line-height:1.55;font-weight:650;margin:0 0 14px;overflow-wrap:anywhere}.original-style{font-size:12px;color:var(--muted);margin:-5px 0 14px;line-height:1.6}.specs{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:12px}.specs span{font-size:12px;background:#f0f4f5;padding:3px 8px;border-radius:5px}code{font-size:12px;color:#6b7b87}details{margin-top:14px;border-top:1px solid #edf1f3;padding-top:10px;font-size:12px;color:var(--muted)}summary{cursor:pointer;color:#395768}details p{max-height:240px;overflow:auto;overflow-wrap:anywhere}.empty{text-align:center;background:white;border:1px dashed var(--line);padding:50px;border-radius:12px}.selection-panel{position:sticky;bottom:0;margin-top:30px;padding:16px 20px;border:1px solid #c2d4d8;border-radius:12px;background:#faffffed;backdrop-filter:blur(12px);box-shadow:0 -5px 24px #172c3b0b}.selection-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}.selection-head strong{font-size:15px}.actions{display:flex;gap:8px;flex-wrap:wrap}.primary{background:var(--accent);color:white;border-color:var(--accent)}.primary:hover{background:#145762;color:white}textarea{width:100%;height:62px;margin-top:12px;padding:8px 12px;resize:vertical;font-size:13px;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink)}#status{font-size:12px;color:var(--muted);margin:5px 0 0}footer{padding-top:0;color:var(--muted);font-size:12px}footer p{margin:5px 0}@media(max-width:1100px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){header,main,footer{padding-left:16px;padding-right:16px}header{padding-top:28px}main{padding-top:12px}.grid{grid-template-columns:1fr;gap:18px}.toolbar{gap:12px}.search{width:100%;flex-basis:100%}.selection-panel{padding:12px;position:static}.hint{font-size:10px}}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body>
<header><div class="eyebrow">EVOWORLD / SAGE SCENE LIBRARY</div><h1>挑选下一间办公室 <span class="count">__COUNT__ 个候选</span></h1><p>先选背景，再放入我们的桌面与 Franka。这里展示的是 SAGE-10k 官方原始场景，图片可点击放大。</p><p>勾选喜欢的场景，复制下方编号发给我即可。当前试装背景已标注，方便对比；画廊可离线使用。</p></header>
<main><div class="toolbar"><input id="search" class="search" type="search" aria-label="搜索场景" placeholder="搜索编号 / 风格：北欧、现代、复古、Art Deco…"><label class="only-selected"><input id="only-selected" type="checkbox">只看已选</label><span id="results-count" class="results-count"></span></div>
<div class="grid">__CARDS__</div><p class="empty" id="empty" hidden>没有匹配的场景，试试其他关键词或取消“只看已选”。</p>
<section class="selection-panel" aria-label="已选场景"><div class="selection-head"><strong id="selection-count">已选 0 个场景</strong><div class="actions"><button id="clear" type="button">清空选择</button><button id="download" type="button">导出清单</button><button id="copy" class="primary" type="button">复制已选编号</button></div></div><textarea id="selection" readonly aria-label="已选场景编号" placeholder="勾选后，编号将显示在这里"></textarea><p id="status" aria-live="polite">选择只保存在此浏览器，不会发送到任何服务器。</p></section></main>
<footer><p>风格描述保留数据集原文。预览用于视觉筛选，不代表物理验证通过；房间尺寸与物品数量来自布局文件。</p><p>本轮仅补充布局与预览，未新增下载完整场景资产。官方来源：<a href="https://huggingface.co/datasets/nvidia/SAGE-10k" target="_blank" rel="noopener">NVIDIA SAGE-10k</a> · Apache-2.0 · 固定候选清单见 <a href="manifest.json">manifest.json</a>。</p></footer>
<script>
'use strict';
const cards=[...document.querySelectorAll('.card')];
const byId=new Map(cards.map(card=>[card.dataset.id,card]));
const storeKey='evoworld-office-gallery-__FINGERPRINT__';
let selected=new Set();
try { const saved=JSON.parse(localStorage.getItem(storeKey)||'[]'); if(Array.isArray(saved)) selected=new Set(saved.filter(id=>byId.has(id))); } catch (_) {}
const search=document.querySelector('#search'), only=document.querySelector('#only-selected'), output=document.querySelector('#selection'), status=document.querySelector('#status');
function selectionText(){return cards.filter(card=>selected.has(card.dataset.id)).map(card=>`候选 ${card.dataset.number} — ${card.dataset.id}`).join('\n');}
function update(){
 let visible=0;const term=search.value.trim().toLocaleLowerCase();
 for(const card of cards){const checked=selected.has(card.dataset.id);card.querySelector('input').checked=checked;card.classList.toggle('selected',checked);card.hidden=!(card.dataset.search.toLocaleLowerCase().includes(term)&&(!only.checked||checked));if(!card.hidden)visible++;}
 document.querySelector('#results-count').textContent=`显示 ${visible} / ${cards.length}`;
 document.querySelector('#empty').hidden=visible>0;
 document.querySelector('#selection-count').textContent=`已选 ${selected.size} 个场景`;
 output.value=selectionText();
 try{localStorage.setItem(storeKey,JSON.stringify([...selected]));}catch(_){}
}
for(const card of cards){
 card.querySelector('input').addEventListener('change',event=>{if(event.target.checked)selected.add(card.dataset.id);else selected.delete(card.dataset.id);update();});
 for(const button of card.querySelectorAll('.view'))button.addEventListener('click',()=>{const img=card.querySelector('.preview');img.src=button.dataset.src;img.alt=`候选 ${card.dataset.number} 官方预览，视角 ${button.dataset.view}`;card.querySelector('.preview-link').href=button.dataset.src;for(const other of card.querySelectorAll('.view')){const active=other===button;other.classList.toggle('active',active);other.setAttribute('aria-pressed',String(active));}});
}
search.addEventListener('input',update);only.addEventListener('change',update);
document.querySelector('#clear').addEventListener('click',()=>{selected.clear();update();status.textContent='已清空选择。';});
document.querySelector('#copy').addEventListener('click',async()=>{if(!selected.size){status.textContent='请先勾选喜欢的场景。';return;}try{await navigator.clipboard.writeText(selectionText());status.textContent='已复制，把编号发给我即可。';}catch(_){output.focus();output.select();status.textContent='编号已选中，请按 Ctrl+C（Mac：Command+C）复制。';}});
document.querySelector('#download').addEventListener('click',()=>{if(!selected.size){status.textContent='请先勾选喜欢的场景。';return;}const url=URL.createObjectURL(new Blob([selectionText()+'\n'],{type:'text/plain;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='EvoWorld-office-selection.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent='已导出场景编号清单。';});
update();
</script></body></html>'''
 fingerprint=hashlib.sha256('\n'.join(r['layout_id'] for r in scenes).encode()).hexdigest()[:12]
 document=template.replace('__COUNT__',str(count)).replace('__CARDS__','\n'.join(cards)).replace('__FINGERPRINT__',fingerprint)
 (out/'index.html').write_text(document)
 manifest={'complete':True,'scene_count':count,'dataset':'nvidia/SAGE-10k','dataset_revision':collection['dataset_revision'],'scope':'official previews and layout metadata; not full dataset, not physics acceptance','excluded_specialty_offices':[{'layout_id':r['layout_id'],'room_types':r['room_types'],'reason':'specialty clinic or postal room, outside general office selection'} for r in excluded],'collection_new_scanned':collection['new_scanned'],'collection_transferred_bytes':collection['transferred_bytes'],'scenes':scenes}
 (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 with (out/'scenes.csv').open('w',newline='',encoding='utf-8-sig') as f:
  keys=['number','layout_id','style','width_m','length_m','height_m','object_count','source_url'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(scenes)
 (out/'README.txt').write_text('EvoWorld office 场景画廊\n\n双击 index.html，用 Firefox / Chrome / Edge 打开。\n勾选喜欢的场景，点击“复制已选编号”，将编号发给助手。\n移动或分享时请保留整个文件夹；也可直接传递同名 ZIP。\n所有预览均为官方原始场景，不包含本项目额外添加的 Franka/GPT 桌面。\n本轮未新增下载完整资产，也没有把视觉候选视为物理验收通过。\n',encoding='utf-8')
 (out/'build_gallery-executed.py').write_bytes(Path(__file__).read_bytes())
 with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for file in sorted(out.rglob('*')):
   if file.is_file():z.write(file,Path(out.name)/file.relative_to(out))
 print(json.dumps({'complete':True,'scenes':count,'html':str(out/'index.html'),'zip_bytes':out.with_suffix('.zip').stat().st_size}),flush=True)

if __name__=='__main__':main()
