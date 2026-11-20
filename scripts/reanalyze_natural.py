"""Re-analyse the archived natural-v02 study with scene-clustered statistics.

Reads only examples/natural-v02/summary.json. The scene is the unit; seeds are
averaged within scene. With three scenes, no CI or test can support a general
claim, and this script makes that explicit instead of hiding it.
"""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.stats import bootstrap_mean, holm, paired, scene_means

OUT = ROOT / 'research/results/reanalysis'
METHODS = ['hold', 'linear', 'flow', 'rife', 'rife_fast']


def main() -> None:
    data = json.loads((ROOT / 'examples/natural-v02/summary.json').read_text())
    report = dict(source='examples/natural-v02/summary.json', unit='scene (seeds averaged within scene)', regimes={})
    lines = ['# Natural-v02 re-analysis: scene-clustered statistics', '',
             'Unit of analysis: scene (3 DAVIS clips); seeds 7/42/99 are averaged within each scene. 95% percentile bootstrap over scenes (10,000 resamples). Paired differences are against two-anchor optical flow.', '']
    for regime in ['scattered', 'long-gap']:
        rows = [r for r in data['rows'] if r['regime'] == regime]
        means = {m: scene_means([r for r in rows if r['method'] == m], 'affected_psnr') for m in METHODS}
        levels = {m: bootstrap_mean(list(v.values())).to_dict() for m, v in means.items()}
        comps = holm([paired(means[m], means['flow'], (m, 'flow')) for m in METHODS if m != 'flow'])
        report['regimes'][regime] = dict(scene_means=means, affected_psnr=levels, vs_flow=[c.to_dict() for c in comps])
        lines += [f'## {regime}', '', '| Method | Affected PSNR (scene mean) | 95% CI | Δ vs flow | 95% CI of Δ | Scenes won | Holm p | CI excludes 0 |', '|---|---:|---|---:|---|---:|---:|---|']
        for m in METHODS:
            lv = levels[m]
            c = next((c for c in comps if c.method == m), None)
            delta = '—' if c is None else f'{c.delta.mean:+.2f}'
            dci = '—' if c is None else f'[{c.delta.low:+.2f}, {c.delta.high:+.2f}]'
            wins = '—' if c is None else f'{c.wins}/{c.delta.n_scenes}'
            hp = '—' if c is None or c.holm_p is None else f'{c.holm_p:.2f}'
            sig = '—' if c is None else ('yes' if c.significant else 'no')
            lines.append(f"| {m} | {lv['mean']:.2f} | [{lv['low']:.2f}, {lv['high']:.2f}] | {delta} | {dci} | {wins} | {hp} | {sig} |")
        lines.append('')
    lines += ['## Reading this honestly', '',
              'With n = 3 scenes, the smallest attainable two-sided Wilcoxon p-value is 0.25, and a bootstrap over three values can only resample those three numbers. These intervals describe *these three clips*; they are not population inferences. Treat this as the template the v1 protocol will fill with many more scenes, not as evidence for or against any method.']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'natural-v02-scene-stats.json').write_text(json.dumps(report, indent=2))
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
