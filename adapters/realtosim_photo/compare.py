"""Diagnostic source/render contact sheets with equal aspect and explicit labels."""
import json
from pathlib import Path
import sys
from PIL import Image, ImageDraw

candidate=Path(sys.argv[1]).resolve();inputs=Path(sys.argv[2]).resolve()
for name in ['main','left','right']:
    source=Image.open(inputs/(name+'.jpg')).convert('RGB')
    render=Image.open(candidate/'preview'/(name+'.png')).convert('RGB')
    source=source.resize(render.size,Image.Resampling.LANCZOS)
    w,h=render.size;canvas=Image.new('RGB',(w*2,h+32),'white')
    canvas.paste(source,(0,32));canvas.paste(render,(w,32))
    draw=ImageDraw.Draw(canvas);draw.text((10,10),name+': USER PHOTO',fill='black');draw.text((w+10,10),'ISAAC RENDER - FITTED CAMERA',fill='black')
    canvas.save(candidate/'preview'/(name+'-comparison.png'))
print('PHOTO_RENDER_COMPARISONS_SAVED')
