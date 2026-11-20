"""Descriptive comparison of published baselines with the v1 methods on the LacunaBench v1 test split.

Merges research/results/lacunabench-v1/rows.csv (test rows) with
research/results/lacunabench-v1-baselines/rows.csv. Same clips, tasks, seeds and
observations. The clip is the unit; paired differences vs RIFE 4.9 are Holm-corrected.
No gate: protocol v2 declares baseline comparisons descriptive.
"""
from pathlib import Path
import csv
import json
import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
from backend.stats import bootstrap_mean, holm, paired, scene_means

OUT = REPO / 'research/results/lacunabench-v1-baselines'
GAPS = (1, 2, 4, 8, 16, 32)
ORDER = ('hold', 'linear', 'flow', 'flow_quadratic', 'rife', 'amt_s', 'ifrnet_s', 'ifrnet', 'amt_s_gopro', 'ifrnet_s_gopro', 'ifrnet_gopro')


def rows():
    out = []
    for path in (REPO / 'research/results/lacunabench-v1/rows.csv', OUT / 'rows.csv', REPO / 'research/results/lacunabench-v1-baselines-gopro/rows.csv'):
        if not path.exists():
            continue
        with path.open() as f:
            out += [r for r in csv.DictReader(f) if r['split'] == 'test']
    return out


def per(sub, m, key):
    vals = [dict(sequence=r['clip'], v=min(float(r[key]), 60.) if key == 'psnr' else float(r[key]))
            for r in sub if r['method'] == m and r[key] not in ('', 'None')]
    return scene_means(vals, 'v') if vals else {}


def main():
    data = rows()
    methods = [m for m in ORDER if any(r['method'] == m for r in data)]
    lines = ['# Published baselines on the LacunaBench v1 test split (11 GoPro clips)', '',
             'AMT-S, IFRNet-S and IFRNet use official weights exported to ONNX (export parity ≤ 3.9e-5). Plain names are the **Vimeo90K-trained (midpoint) checkpoints**, which reproduce their published Vimeo90K PSNR within 0.2 dB (see `../vimeo-parity/`). `*_gopro` are the official **arbitrary-time GoPro ×8 checkpoints** added by protocol v2 amendment 1. Caveat: they were trained on GOPRO train scenes, and this test split uses GOPRO test scenes from the same camera. All methods received identical observations. The clip is the unit. Comparisons are descriptive (protocol v2).', '',
             '## PSNR (dB) by gap length', '', '| Gap | ' + ' | '.join(methods) + ' |', '|---|' + '---:|' * len(methods)]
    report = dict(psnr={}, lpips={}, vs_rife={})
    for g in GAPS:
        sub = [r for r in data if r['task'] == f'gap{g}']
        vals = {m: np.mean(list(per(sub, m, 'psnr').values())) for m in methods}
        best = max(vals, key=vals.get)
        report['psnr'][g] = vals
        lines.append(f'| {g} | ' + ' | '.join(f"**{v:.2f}**" if m == best else f'{v:.2f}' for m, v in vals.items()) + ' |')
    lines += ['', '## LPIPS by gap length (lower is better)', '', '| Gap | ' + ' | '.join(methods) + ' |', '|---|' + '---:|' * len(methods)]
    for g in GAPS:
        sub = [r for r in data if r['task'] == f'gap{g}']
        vals = {m: np.mean(list(per(sub, m, 'lpips').values())) for m in methods}
        best = min(vals, key=vals.get)
        report['lpips'][g] = vals
        lines.append(f'| {g} | ' + ' | '.join(f"**{v:.3f}**" if m == best else f'{v:.3f}' for m, v in vals.items()) + ' |')
    lines += ['', '## Loss processes at 25% (PSNR dB / LPIPS)', '', '| Process | ' + ' | '.join(methods) + ' |', '|---|' + '---:|' * len(methods)]
    for proc in ('bernoulli', 'gilbert_elliott', 'gop_chain'):
        sub = [r for r in data if r['task'] == proc]
        lines.append(f'| {proc} | ' + ' | '.join(f"{np.mean(list(per(sub, m, 'psnr').values())):.2f} / {np.mean(list(per(sub, m, 'lpips').values())):.3f}" for m in methods) + ' |')
    for label, sel in [('all contiguous gaps', lambda r: r['process'] == 'contiguous'),
                       ('long gaps (≥ 8)', lambda r: r['process'] == 'contiguous' and int(r['task'][3:]) >= 8)]:
        sub = [r for r in data if sel(r)]
        rife = per(sub, 'rife', 'psnr')
        comps = holm([paired(per(sub, m, 'psnr'), rife, (m, 'rife')) for m in methods if m != 'rife'])
        report['vs_rife'][label] = [c.to_dict() for c in comps]
        lines += ['', f'## ΔPSNR vs RIFE 4.9, {label} (paired over clips, Holm-corrected)', '', '| Method | Δ (dB) | 95% CI | Clips better | Holm p |', '|---|---:|---|---:|---:|']
        lines += [f'| {c.method} | {c.delta.mean:+.2f} | [{c.delta.low:+.2f}, {c.delta.high:+.2f}] | {c.wins}/{c.delta.n_scenes} | {"—" if c.holm_p is None else f"{c.holm_p:.4f}"} |' for c in comps]
    secs = {m: np.mean([float(r['seconds_per_frame']) for r in data if r['method'] == m]) for m in methods}
    report['seconds_per_frame'] = secs
    lines += ['', 'CPU seconds per missing frame at 640×360 (indicative; runs shared the CPU): ' + ', '.join(f'{m} {s:.2f}' for m, s in secs.items()) + '.']
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2, default=float))
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
