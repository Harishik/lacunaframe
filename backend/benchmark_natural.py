"""Natural footage study: 3 sequences × 2 gap regimes × 3 seeds; no training."""
import csv
import json
import logging
import os
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi.testclient import TestClient
from .app import app, ROOT, NATURAL_IDS
logging.getLogger('httpx').setLevel(logging.WARNING)
OUT=Path(__file__).resolve().parents[1]/'examples/natural-v02'
METHODS=['hold','linear','flow','rife','rife_fast']

def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows=[]
    with TestClient(app) as c:
        for source in NATURAL_IDS:
            for regime in ['scattered','long-gap']:
                for seed in [7,42,99]:
                    config=dict(source=source,seed=seed,drop=.2 if regime=='scattered' else .1,gap=0 if regime=='scattered' else 6,start=.4,occlusion=0,noise=0,blur=0)
                    response=c.post('/api/scenarios',json=config);response.raise_for_status();scenario=response.json()
                    response=c.post('/api/run',json={'scenario':scenario['id'],'methods':METHODS});response.raise_for_status();job=response.json()
                    while job['status']=='running':
                        time.sleep(.2);job=c.get('/api/results/'+job['id']).json()
                    assert job['status']=='complete',job
                    label=f'{source}-{regime}-{seed}';dest=OUT/label;dest.mkdir(exist_ok=True)
                    p=ROOT/job['id'];report=job['report']
                    for file in p.iterdir():
                        if file.suffix in ('.json','.csv','.mp4','.gif'):shutil.copyfile(file,dest/file.name)
                    if seed==42:
                        ref=np.load(p/'reference.npy');missing=np.load(p/'missing.npy')
                        out=np.load(p/'rife.npy')
                        errors=np.mean((out.astype(float)-ref.astype(float))**2,axis=(1,2,3));errors[~missing]=-1
                        index=int(errors.argmax())
                        labels=['Reference','Damaged',*METHODS]
                        frames=[ref[index],np.load(p/'corrupted.npy')[index]]+[np.load(p/f'{m}.npy')[index] for m in METHODS]
                        fig,axes=plt.subplots(2,4,figsize=(16,6.5))
                        for ax,title,frame in zip(axes.flat,labels,frames):ax.imshow(frame);ax.set_title(title);ax.axis('off')
                        axes.flat[-1].axis('off');fig.suptitle(f'{source} · {regime} · seed {seed} · worst RIFE missing frame {index} (same frame for all methods)')
                        fig.tight_layout();fig.savefig(dest/'comparison.png',dpi=120);plt.close(fig)
                        fig,ax=plt.subplots(figsize=(10,3))
                        for m,metric in report['methods'].items():ax.plot([r['time'] for r in metric['frames']],[r['mse'] for r in metric['frames']],label=m)
                        ax.set(xlabel='Time (s)',ylabel='RGB MSE',title=label);ax.legend();fig.tight_layout();fig.savefig(dest/'error-curve.png');plt.close(fig)
                    for method,metric in report['methods'].items():
                        rows.append(dict(sequence=source,regime=regime,seed=seed,method=method,count=scenario['count'],missing=sum(scenario['missing']),psnr=metric['psnr'],affected_psnr=metric['affected_psnr'],affected_ssim=metric['affected_ssim'],temporal_error=metric['temporal_error'],runtime_seconds=metric['runtime_seconds'],seconds_per_missing_frame=metric['runtime_seconds']/sum(scenario['missing']),process_peak_rss_mib=metric['process_peak_rss_mib'],model_initialization_seconds=metric['model_initialization_seconds']))
                    # Free only this benchmark's temporary runs; portable results already saved.
                    shutil.rmtree(p);shutil.rmtree(ROOT/scenario['id'])
                    print(label,'complete',flush=True)
    summary=[]
    for regime in ['scattered','long-gap']:
        for method in METHODS:
            subset=[r for r in rows if r['regime']==regime and r['method']==method]
            summary.append(dict(regime=regime,method=method,n=len(subset),mean_affected_psnr=float(np.mean([r['affected_psnr'] for r in subset])),mean_affected_ssim=float(np.mean([r['affected_ssim'] for r in subset])),mean_seconds_per_missing_frame=float(np.mean([r['seconds_per_missing_frame'] for r in subset]))))
    data=dict(protocol='3 natural DAVIS sequences × 2 temporal corruption regimes × 3 seeds. Equal-weight averages of per-experiment scores; not pooled dataset PSNR. No training or tuning on these targets. MP4 mirror is a lossy reference; no claim of unseen model training data.',rows=rows,summary=summary)
    (OUT/'summary.json').write_text(json.dumps(data,indent=2))
    with (OUT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
