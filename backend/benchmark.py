"""Run through the real API; export portable evidence, independent of temp storage."""
import json
import shutil
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi.testclient import TestClient
from .app import app, ROOT

OUT=Path(__file__).resolve().parents[1]/'examples/synthetic-v01'
OUT.mkdir(exist_ok=True,parents=True)
summary=[]
with TestClient(app) as c:
    for i in range(3):
        config=dict(source=f'demo{i}',seed=42,drop=.15,gap=[2,8,0][i],occlusion=.25 if i==2 else 0,noise=0,blur=0)
        s=c.post('/api/scenarios',json=config).json()
        j=c.post('/api/run',json={'scenario':s['id'],'methods':['hold','linear','flow','telea']}).json()
        while j['status']=='running':
            time.sleep(.2);j=c.get('/api/results/'+j['id']).json()
        assert j['status']=='complete',j
        dest=OUT/f'scenario-{i+1}';dest.mkdir(exist_ok=True)
        for path in (ROOT/j['id']).iterdir():
            if path.suffix in ('.json','.csv','.mp4','.gif'):shutil.copy(path,dest/path.name)
        for path in (ROOT/f'demo{i}').glob('original.*'):shutil.copy(path,dest/path.name)
        for kind in ('original','corrupted','flow','error','mask'):
            (dest/f'{kind}-frame20.png').write_bytes(c.get(f"/api/frames/{j['id']}/{kind}/20").content)
        report=j['report'];fig,ax=plt.subplots(figsize=(9,3))
        for method,metric in report['methods'].items():
            ax.plot([r['time'] for r in metric['frames']],[r['mse'] for r in metric['frames']],label=method)
            summary.append(dict(scenario=i+1,method=method,psnr=metric['psnr'],affected_psnr=metric['affected_psnr'],ssim=metric['ssim'],runtime_seconds=metric['runtime_seconds']))
        ax.set(xlabel='Time (s)',ylabel='MSE (RGB, 0–255)',title=f'Synthetic scenario {i+1} · seed 42');ax.legend();fig.tight_layout();fig.savefig(dest/'error-curve.png');plt.close(fig)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
