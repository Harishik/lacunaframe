"""Run a fixed, identical-pixel 4.9 vs 4.26 comparison on all 18 conditions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import json
import time
import numpy as np
from fastapi.testclient import TestClient
from backend.app import app,ROOT,NATURAL_IDS,encode
from backend.core import corrupt,evaluate,restore
from backend.candidate import interpolate,session,INFO
from backend.learned import session as baseline_session,model_info

OUT=Path(__file__).resolve().parents[1]/'examples/checkpoint-v03'

def main() -> None:
    OUT.mkdir(exist_ok=True,parents=True)
    start=time.perf_counter();session();baseline_session();initialization=time.perf_counter()-start
    rows=[]
    with TestClient(app):
        for source in NATURAL_IDS:
            ref=np.load(ROOT/source/'reference.npy')
            fps=json.loads((ROOT/source/'meta.json').read_text())['fps']
            for regime in ['scattered','long-gap']:
                for seed in [7,42,99]:
                    config=dict(seed=seed,drop=.2 if regime=='scattered' else .1,gap=0 if regime=='scattered' else 6,start=.4,occlusion=0,noise=0,blur=0)
                    damaged,missing,mask=corrupt(ref,config);affected=np.broadcast_to(missing[:,None,None],mask.shape)
                    valid=np.flatnonzero(~missing)
                    label=f'{source}-{regime}-{seed}';dest=OUT/label;dest.mkdir(exist_ok=True)
                    report=dict(source=source,config=config,missing=missing.tolist(),fps=fps,methods={})
                    for method in ['rife49','rife426']:
                        start=time.perf_counter()
                        if method=='rife49': out,_,_=restore(damaged,missing,mask,'rife')
                        else:
                            out=damaged.copy()
                            for i in np.flatnonzero(missing):
                                left=valid[valid<i];right=valid[valid>i]
                                if not len(left) or not len(right):out[i]=damaged[valid[np.argmin(abs(valid-i))]]
                                else:
                                    l,r=int(left[-1]),int(right[0]);out[i]=interpolate(damaged[l],damaged[r],(i-l)/(r-l))
                        runtime=time.perf_counter()-start
                        assert np.array_equal(out[~missing],ref[~missing])
                        metric=evaluate(ref,out,affected,fps)
                        report['methods'][method]=dict(**metric,runtime_seconds=runtime,model=model_info() if method=='rife49' else INFO)
                        rows.append(dict(sequence=source,regime=regime,seed=seed,method=method,affected_psnr=metric['affected_psnr'],affected_ssim=metric['affected_ssim'],temporal_error=metric['temporal_error'],runtime_seconds=runtime,seconds_per_missing_frame=runtime/int(missing.sum())))
                        if seed==42:encode(dest,method,out,fps)
                    (dest/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
                    print(label,[(m,round(v['affected_psnr'],3)) for m,v in report['methods'].items()],flush=True)
    summary=[]
    for regime in ['scattered','long-gap']:
        for method in ['rife49','rife426']:
            subset=[r for r in rows if r['regime']==regime and r['method']==method]
            summary.append(dict(regime=regime,method=method,n=len(subset),**{f'mean_{k}':float(np.mean([r[k] for r in subset])) for k in ['affected_psnr','affected_ssim','seconds_per_missing_frame']}))
    data=dict(protocol='Identical normalized input arrays, corruption and metric implementation. 3 DAVIS clips × 2 gap regimes × seeds 7/42/99. CPU, 2 ORT threads, warm inference timed separately. No retraining. Candidate chosen after the original benchmark: exploratory comparison, not a held-out test.',combined_initialization_seconds=initialization,rows=rows,summary=summary)
    (OUT/'summary.json').write_text(json.dumps(data,indent=2))
    with (OUT/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
