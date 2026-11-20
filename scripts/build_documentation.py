"""Build documentation figures/GIFs from real aligned arrays and measured reports.

No stock media or fabricated scores. Recomputes the documented CPU example,
then annotates actual frames for a readable, reproducible visual narrative.
"""
from pathlib import Path
import json
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from backend.core import restore,evaluate
from backend import decode
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/visuals'
OUT.mkdir(parents=True,exist_ok=True)

def font(size: int):
    path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default(size=size)

def main() -> None:
    pack=ROOT/'examples/external-pack-v03'
    with np.load(pack/'inputs.npz') as d:
        damaged,missing,mask,fps=d['frames'],d['missing'],d['mask'],float(d['fps'])
    with np.load(pack/'targets.npz') as d:reference,affected=d['frames'],d['affected']
    outputs={m:restore(damaged,missing,mask,m)[0] for m in ['linear','flow','rife']}
    metrics={m:evaluate(reference,v,affected,fps) for m,v in outputs.items()}
    error=((outputs['rife'].astype(float)-reference.astype(float))**2).mean(axis=(1,2,3))
    error[~missing]=-1;index=int(error.argmax())
    (OUT/'example.json').write_text(json.dumps(dict(source='natural-bmx-trees',seed=42,regime='long-gap',fps=fps,count=len(reference),worst_rife_frame=index,missing=missing.tolist(),metrics=metrics),indent=2,allow_nan=False))
    frames=[]
    for i in range(len(reference)):
        canvas=Image.new('RGB',(960,236),'#17191c');draw=ImageDraw.Draw(canvas)
        for k,(title,a) in enumerate([('01 / Reference',reference),('02 / Damaged',damaged),('03 / RIFE 4.9',outputs['rife'])]):
            draw.text((k*320+10,9),title,font=font(14),fill='#eeeae2')
            canvas.paste(Image.fromarray(a[i]),(k*320,34))
        state='MISSING — estimate is unverified' if missing[i] else 'OBSERVED — pixels preserved'
        draw.text((10,219),f'Frame {i:02d} / {len(reference)-1}    {state}',font=font(11),fill='#ee9278' if missing[i] else '#c6cbd3')
        if missing[i]:draw.rectangle((320,34,639,213),outline='#ee9278',width=2)
        frames.append(canvas)
    frames[0].save(OUT/'recovery-loop.gif',save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0,optimize=True)
    for label,array in [('reference',reference),('damaged',damaged),*outputs.items()]:
        Image.fromarray(array[index]).save(OUT/f'worst-{label}.png')
    fig,axes=plt.subplots(2,3,figsize=(12,5.7),facecolor='#17191c')
    values=[('Reference',reference[index]),('Damaged (explicitly absent)',damaged[index]),('Linear',outputs['linear'][index]),('Optical flow',outputs['flow'][index]),('RIFE 4.9',outputs['rife'][index])]
    for ax,(title,a) in zip(axes.flat,values):ax.imshow(a);ax.set_title(title,color='#eeeae2',fontsize=12);ax.axis('off')
    heat=np.abs(outputs['rife'][index].astype(float)-reference[index]).mean(-1)
    im=axes.flat[-1].imshow(heat,cmap='inferno',vmin=0,vmax=255);axes.flat[-1].set_title('RIFE absolute RGB error · 0–255',color='#eeeae2',fontsize=12);axes.flat[-1].axis('off')
    fig.suptitle(f'The hardest absent frame for RIFE · frame {index} · BMX / six-frame gap / seed 42',color='#eeeae2',fontsize=14)
    fig.tight_layout();fig.savefig(OUT/'failure-contact-sheet.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
    data=json.loads((ROOT/'examples/natural-v02/summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),facecolor='#eeeae2')
    names={'hold':'Nearest','linear':'Linear','flow':'Flow','rife':'RIFE 4.9','rife_fast':'RIFE fast'}
    for ax,regime in zip(axes,['scattered','long-gap']):
        rows=[r for r in data['summary'] if r['regime']==regime]
        ax.set_facecolor('#eeeae2')
        bars=ax.barh([names[r['method']] for r in rows],[r['mean_affected_psnr'] for r in rows],color=['#8b939e' if not r['method'].startswith('rife') else '#b96146' for r in rows])
        ax.bar_label(bars,fmt='%.2f',padding=5);ax.invert_yaxis();ax.set_xlim(0,24);ax.set_title(regime);ax.set_xlabel('Affected-frame PSNR · dB ↑');ax.spines[['top','right','left']].set_visible(False)
    fig.suptitle('Measured recovery on recorded footage\n3 DAVIS scenes × 3 seeds per regime · equal-weight means, not paper benchmark scores',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'natural-results.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,3.5),facecolor='#eeeae2');ax.set_facecolor('#eeeae2')
    for m,color in [('linear','#969da7'),('flow','#5b7f9f'),('rife','#b96146')]:
        ax.plot(np.arange(len(reference))/fps,[r['mse'] for r in metrics[m]['frames']],label=m,color=color)
    for i in np.flatnonzero(missing):ax.axvspan((i-.5)/fps,(i+.5)/fps,color='#ee9278',alpha=.13)
    ax.set(title='Errors occur at missing samples · shaded intervals are absent observations',xlabel='Time · seconds',ylabel='RGB MSE ↓');ax.legend();ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(OUT/'error-timeline.png',dpi=150);plt.close(fig)
    # A short synthetic control remains visibly labeled as such.
    synthetic=[]
    for a in decode.frames(ROOT/'examples/synthetic-v01/scenario-1/flow.mp4'):
        image=Image.new('RGB',(320,208),'#17191c');image.paste(Image.fromarray(a),(0,28));ImageDraw.Draw(image).text((8,7),'SYNTHETIC CONTROL / optical flow',font=font(11),fill='#eeeae2');synthetic.append(image)
    synthetic[0].save(OUT/'synthetic-control.gif',save_all=True,append_images=synthetic[1:],duration=83,loop=0,optimize=True)
    guide=ROOT/'docs/experience/index.html'
    data=dict(natural=data,checkpoints=json.loads((ROOT/'examples/checkpoint-v03/summary.json').read_text()),example=json.loads((OUT/'example.json').read_text()))
    guide.write_text(re.sub(r'(<script id="study-data" type="application/json">).*?(</script>)',lambda m:m[1]+json.dumps(data,separators=(',',':'))+m[2],guide.read_text(),flags=re.S))
    print(json.dumps({'visuals':len(list(OUT.iterdir())),'worst_frame':index,'gif_bytes':(OUT/'recovery-loop.gif').stat().st_size}))

if __name__=='__main__':main()
