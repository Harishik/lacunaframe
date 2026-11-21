"""Oracle test of v2's motion pipeline, independent of learning.

Feeds classical Farneback pair flows (the same flows as the 'two-anchor flow' baseline) into
v2's own trajectory -> reversal -> warp -> blend path at quarter resolution, with the learned
correction and refinement switched off. If this oracle beats linear blending by a wide margin,
the pipeline is sound and the bottleneck is how the pair flows are learned. Dev clips only.

    ../.venv-train/bin/python research/lacunanet/oracle_pipeline.py [clips]
"""
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch, torch.nn.functional as F
import cv2
FARNEBACK = (.5, 3, 21, 4, 7, 1.5, 0)   # same parameters as backend.core (training env lacks skimage)


def flow_between(a, b):
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(ga, gb, None, *FARNEBACK), cv2.calcOpticalFlowFarneback(gb, ga, None, *FARNEBACK)
from research.lacunanet.model_v2 import trajectory, reverse, warp
from research.lacunanet.classical_flow import lk_flow
from research.lacunabench.recoverability import brackets
from research.scripts.lacunabench_v1 import tasks

EXT = REPO.parent / 'lacuna-data'
ps = lambda e, g: 10 * np.log10(255 ** 2 / max(np.mean((np.clip(e, 0, 255).round() - g) ** 2), 1e-9))
T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32).transpose(2, 0, 1)[None])


def synth(f0, f1, f01, f10, t, quarter=True, prequarter=False):
    """v2 path: constant velocity (no outer anchors), reversal, warp, time-linear mask."""
    a0, a1 = T(f0) / 255., T(f1) / 255.
    F01, F10 = T(f01), T(f10)
    size = a0.shape[-2:]
    if quarter and not prequarter:
        q = (size[0] // 4, size[1] // 4)
        F01 = F.interpolate(F01, size=q, mode='area') / 4; F10 = F.interpolate(F10, size=q, mode='area') / 4
    tt, one = torch.tensor([t]), torch.ones(1)
    d0 = trajectory(F01, torch.zeros_like(F01), one, one, tt, torch.zeros(1))
    d1 = F10 * (1 - t)
    b0, b1 = reverse(d0), reverse(d1)
    if quarter:
        b0 = F.interpolate(b0, size=size, mode='bilinear', align_corners=False) * 4
        b1 = F.interpolate(b1, size=size, mode='bilinear', align_corners=False) * 4
    out = (1 - t) * warp(a0, b0) + t * warp(a1, b1)
    return out[0].numpy().transpose(1, 2, 0) * 255


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    torch.set_num_threads(2)
    splits = json.loads((REPO / 'research/protocols/splits-v1.json').read_text())
    res = []
    for c in splits['dev'][:n]:
        frames = np.load(EXT / 'snu-film' / c['file'])['frames'][:, :352]
        for task in tasks(c['id'], len(frames)):
            if task['task'] not in ('gap1', 'gap2', 'gap4', 'gap8'):
                continue
            for left, right, l2, r2, idx in brackets(task['missing']):
                f01, f10 = flow_between(frames[left], frames[right])
                A, B = T(frames[left]) / 255., T(frames[right]) / 255.
                l01 = lk_flow(A, B)[0].numpy().transpose(1, 2, 0); l10 = lk_flow(B, A)[0].numpy().transpose(1, 2, 0)
                q01 = lk_flow(A, B, out_scale=4)[0].numpy().transpose(1, 2, 0); q10 = lk_flow(B, A, out_scale=4)[0].numpy().transpose(1, 2, 0)
                for i in idx:
                    t = (i - left) / (right - left)
                    lin = (1 - t) * frames[left].astype(float) + t * frames[right]
                    res.append(dict(task=task['task'], linear=ps(lin, frames[i]),
                                    oracle_quarter=ps(synth(frames[left], frames[right], f01, f10, t, True), frames[i]),
                                    oracle_full=ps(synth(frames[left], frames[right], f01, f10, t, False), frames[i]),
                                    lk_quarter=ps(synth(frames[left], frames[right], l01, l10, t, True), frames[i]),
                                    lk_at_quarter=ps(synth(frames[left], frames[right], q01, q10, t, True, True), frames[i])))
    out = {}
    for task in ('gap1', 'gap2', 'gap4', 'gap8'):
        sub = [r for r in res if r['task'] == task]
        out[task] = {k: round(float(np.mean([r[k] for r in sub])), 2) for k in ('linear', 'oracle_quarter', 'oracle_full', 'lk_quarter', 'lk_at_quarter')} | dict(n=len(sub))
    (REPO / 'research/results/lacunanet/oracle-pipeline.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
