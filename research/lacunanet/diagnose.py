"""Diagnose where a LacunaNet checkpoint fails (dev split only; training environment).

For every missing frame of the first N dev clips (all T3 tasks) it reconstructs with the
checkpoint, then joins per-frame PSNR of linear / flow / RIFE 4.9 from the stored v1 rows and
stratifies by inner-anchor displacement (median Farneback magnitude at 320x180, x2 for 640 px)
and by gap length. Output: research/results/lacunanet/diagnosis-<ckpt>.json

    ../.venv-train/bin/python research/lacunanet/diagnose.py <ckpt.pt> [n_clips]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import cv2
import numpy as np
import torch
from research.lacunanet.model import LacunaNet, make_cond
from research.lacunabench.recoverability import brackets
from research.scripts.lacunabench_v1 import tasks

EXT = REPO.parent / 'lacuna-data'
torch.set_grad_enabled(False)


def main():
    ck, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 6
    st = torch.load(ck, map_location='cpu')
    m = LacunaNet(st['c']).eval(); m.load_state_dict(st['model'])
    torch.set_num_threads(6)
    splits = json.loads((REPO / 'research/protocols/splits-v1.json').read_text())
    rows = {}
    with (REPO / 'research/results/lacunabench-v1/rows.csv').open() as f:
        for r in csv.DictReader(f):
            if r['split'] == 'dev':
                rows[(r['clip'], r['task'], int(r['frame']), r['method'])] = min(float(r['psnr']), 60.)
    out = []
    t = lambda x: torch.from_numpy(x.astype(np.float32).transpose(2, 0, 1)[None] / 255.)
    for c in splits['dev'][:n]:
        frames = np.load(EXT / 'snu-film' / c['file'])['frames']
        for task in tasks(c['id'], len(frames)):
            for left, right, l2, r2, idx in brackets(task['missing']):
                ga = cv2.cvtColor(cv2.resize(frames[left], (320, 180)), cv2.COLOR_RGB2GRAY)
                gb = cv2.cvtColor(cv2.resize(frames[right], (320, 180)), cv2.COLOR_RGB2GRAY)
                disp = float(np.median(np.linalg.norm(cv2.calcOpticalFlowFarneback(ga, gb, None, .5, 4, 15, 3, 5, 1.2, 0), axis=-1))) * 2
                gap = right - left - 1
                for i in idx:
                    tt = (i - left) / (right - left)
                    span = right - left
                    cond = torch.tensor([make_cond(tt, gap, -tt - (left - l2) / span if l2 is not None else 0, 1 - tt + (r2 - right) / span if r2 is not None else 0, l2 is not None, r2 is not None)])
                    z = torch.zeros_like(t(frames[left]))
                    f, s = m(t(frames[left]), t(frames[right]), t(frames[l2]) if l2 is not None else z, t(frames[r2]) if r2 is not None else z, cond)
                    est = np.clip(f[0].numpy().transpose(1, 2, 0) * 255, 0, 255).round()
                    mse = float(np.mean((est - frames[i].astype(float)) ** 2))
                    rec = dict(clip=c['id'], task=task['task'], gap=gap, disp=disp, lacunanet=min(60., 10 * np.log10(255 ** 2 / max(mse, 1e-12))))
                    for meth in ('linear', 'flow', 'rife'):
                        rec[meth] = rows.get((c['id'], task['task'], int(i), meth))
                    out.append(rec)
        print('clip', c['id'], len(out), flush=True)
    def summarize(key, edges):
        res = {}
        for lo, hi in edges:
            sub = [r for r in out if lo <= r[key] < hi and r['rife'] is not None]
            if sub:
                res[f'{lo}-{hi}'] = dict(n=len(sub), **{mm: round(float(np.mean([r[mm] for r in sub])), 2) for mm in ('linear', 'flow', 'rife', 'lacunanet')})
        return res
    report = dict(checkpoint=ck.name, frames=len(out),
                  by_displacement_px=summarize('disp', [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]),
                  by_gap=summarize('gap', [(1, 2), (2, 3), (3, 5), (5, 9), (9, 17), (17, 33)]))
    path = REPO / 'research/results/lacunanet' / f'diagnosis-{ck.stem}.json'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
