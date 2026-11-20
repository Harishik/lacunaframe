"""Diagnostic: quadratic multi-anchor flow vs two-anchor flow on archived DAVIS conditions.

Protocol v1 treats DAVIS as diagnostic only, so this is NOT the G1 gate (which
needs the frozen test split). It reproduces the exact natural-v02 conditions,
first checks that two-anchor flow matches the archived scores, then compares
methods with scene-clustered statistics. It also adds a synthetic control with
known nonlinear (sinusoidal) motion, where a quadratic model should help.
"""
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from backend import decode
from backend.app import normalize
from backend.core import corrupt, demo, evaluate, restore
from backend.stats import bootstrap_mean, paired, scene_means

OUT = ROOT / 'research/results/multianchor-diagnostic'
SCENES = ['blackswan', 'bmx-trees', 'breakdance']
GAPS = {'scattered': (.2, 0), 'long-gap': (.1, 6)}
SEEDS = [7, 42, 99]


def reference(scene: str) -> np.ndarray:
    frames, _ = decode.sampled(ROOT / f'assets/video/{scene}.mp4', every=2, limit=48)
    return np.stack([normalize(a) for a in frames])


def score(ref, method, cfg):
    damaged, missing, mask = corrupt(ref, cfg)
    affected = mask.copy()
    affected[missing] = True
    began = time.perf_counter()
    out, _, _ = restore(damaged, missing, mask, method)
    elapsed = time.perf_counter() - began
    return evaluate(ref, out, affected, 12.)['affected_psnr'], elapsed / max(1, int(missing.sum()))


def main() -> None:
    archived = json.loads((ROOT / 'examples/natural-v02/summary.json').read_text())['rows']
    rows, parity = [], []
    for scene in SCENES:
        ref = reference(scene)
        for regime, (drop, gap) in GAPS.items():
            for seed in SEEDS:
                cfg = dict(seed=seed, drop=drop, gap=gap, start=.4, occlusion=0, noise=0, blur=0)
                for method in ('flow', 'flow_quadratic'):
                    psnr, spf = score(ref, method, cfg)
                    rows.append(dict(sequence=scene, regime=regime, seed=seed, method=method, affected_psnr=psnr, seconds_per_missing_frame=spf))
                    if method == 'flow':
                        old = next(r for r in archived if r['sequence'] == f'natural-{scene}' and r['regime'] == regime and r['seed'] == seed and r['method'] == 'flow')
                        parity.append(abs(old['affected_psnr'] - psnr))
    # Synthetic control: sinusoidal vertical motion, 6-frame gaps, three seeds per scene.
    for kind in (0, 1, 2):
        ref = demo(kind)
        for seed in SEEDS:
            cfg = dict(seed=seed, drop=.1, gap=6, start=.4, occlusion=0, noise=0, blur=0)
            for method in ('flow', 'flow_quadratic'):
                psnr, spf = score(ref, method, cfg)
                rows.append(dict(sequence=f'synthetic-{kind}', regime='synthetic-long-gap', seed=seed, method=method, affected_psnr=psnr, seconds_per_missing_frame=spf))

    report = dict(status='diagnostic only (DAVIS + synthetic); not the G1 gate', max_flow_parity_error_db=max(parity), groups={})
    lines = ['# Multi-anchor flow: diagnostic check', '',
             'This is **not** the pre-registered G1 gate: DAVIS is diagnostic-only under protocol v1, and G1 needs the frozen test split. Unit of analysis: scene, with seeds averaged within scene.', '',
             f'Parity check: two-anchor flow reproduces the archived natural-v02 scores (max |Δ| = {max(parity):.2e} dB).', '',
             '| Condition | Flow (dB) | Quadratic (dB) | Δ (scene mean) | 95% CI of Δ | Scenes won | s / missing frame (flow → quad) |', '|---|---:|---:|---:|---|---:|---|']
    for regime in ['scattered', 'long-gap', 'synthetic-long-gap']:
        sub = [r for r in rows if r['regime'] == regime]
        f = scene_means([r for r in sub if r['method'] == 'flow'], 'affected_psnr')
        q = scene_means([r for r in sub if r['method'] == 'flow_quadratic'], 'affected_psnr')
        c = paired(q, f, ('flow_quadratic', 'flow'))
        tf = np.mean([r['seconds_per_missing_frame'] for r in sub if r['method'] == 'flow'])
        tq = np.mean([r['seconds_per_missing_frame'] for r in sub if r['method'] == 'flow_quadratic'])
        report['groups'][regime] = dict(flow=f, quadratic=q, comparison=c.to_dict(), flow_level=bootstrap_mean(list(f.values())).to_dict())
        lines.append(f"| {regime} | {np.mean(list(f.values())):.3f} | {np.mean(list(q.values())):.3f} | {c.delta.mean:+.3f} | [{c.delta.low:+.3f}, {c.delta.high:+.3f}] | {c.wins}/{c.delta.n_scenes} | {tf:.3f} → {tq:.3f} |")
    report['rows'] = rows
    lines += ['', 'With three scenes per group these intervals describe these clips only. The decision on multi-anchor reconstruction waits for G1 on the test split.']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
