"""LacunaBench v1: task T3 (missing runs) on the frozen SNU-FILM dev/test splits.

For every clip: six single contiguous gaps (g = 1, 2, 4, 8, 16, 32) and three
loss processes at a 25% loss rate (Bernoulli, Gilbert-Elliott burst 4, GOP-8
chain). Five reconstructors receive identical observations. Per missing frame
it records PSNR, SSIM and LPIPS, plus DISTS on scheduled test frames (protocol
v1, amendment 1), and stores 80x45 float16 absolute-error maps outside git for
C2/C3. Seeds derive from clip and task names, so reruns are identical.

    .venv/bin/python research/scripts/lacunabench_v1.py run [--workers 3] [--clips N]
    .venv/bin/python research/scripts/lacunabench_v1.py analyse
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np

EXTERNAL = Path(os.environ.get('LACUNA_EXTERNAL', REPO.parent / 'lacuna-data')).resolve()
SPLITS = json.loads((REPO / 'research/protocols/splits-v1.json').read_text())
DATASET = os.environ.get('LACUNA_BENCH_DATASET', 'snu')   # 'snu' (v1 dev/test) or 'xiph' (v2 confirmation)
SUFFIX = os.environ.get('LACUNA_BENCH_TAG', '' if DATASET == 'snu' else f'-{DATASET}')  # separate output folder per run family
RUN_SPLITS = tuple(os.environ.get('LACUNA_BENCH_SPLITS', 'dev,test').split(','))
OUT = REPO / f'research/results/lacunabench-v1{SUFFIX}'
MAPS = EXTERNAL / f'results/lacunabench-v1{SUFFIX}'
DATA_DIR = {'snu': 'snu-film', 'xiph': 'xiph', 'xiph-c': 'xiph-c'}[DATASET]
GAPS = (1, 2, 4, 8, 16, 32)
METHODS = tuple(os.environ.get('LACUNA_BENCH_METHODS', 'hold,linear,flow,flow_quadratic,rife').split(','))
RATE = .25
DISTS_GAPS = (4, 16)
MAP_SIZE = (80, 45)


def seed(*parts) -> int:
    return zlib.crc32('/'.join(map(str, parts)).encode())


def tasks(clip: str, n: int) -> list[dict]:
    from research.lacunabench import losses
    out = []
    for g in GAPS:
        rng = np.random.default_rng(seed(clip, 'gap', g))
        start = int(rng.integers(2, n - 2 - g + 1))
        out.append(dict(task=f'gap{g}', process='contiguous', missing=losses.contiguous(n, start, g)))
    q = losses.gop_frame_loss_for_rate(RATE, 8)
    procs = {'bernoulli': lambda r: losses.bernoulli(n, RATE, r),
             'gilbert_elliott': lambda r: losses.gilbert_elliott(n, RATE, 4, r),
             'gop_chain': lambda r: losses.gop_chain(n, q, 8, r)}
    for name, fn in procs.items():
        out.append(dict(task=name, process=name, missing=fn(np.random.default_rng(seed(clip, name)))))
    return out


def run_lengths(missing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each sample: length of the missing run containing it, and position t in (0,1) within its bracket."""
    n = len(missing)
    length, position = np.zeros(n, int), np.zeros(n)
    i = 0
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            for k in range(i, j):
                length[k] = j - i
                position[k] = (k - (i - 1)) / (j - (i - 1))
            i = j
        else:
            i += 1
    return length, position


def clip_worker(args: tuple[str, str, str]) -> list[dict]:
    clip, split, file = args
    os.environ.setdefault('LACUNA_CPU_THREADS', '2')
    import cv2
    from skimage.metrics import structural_similarity
    from backend import perceptual
    from backend.core import restore
    frames = np.load(EXTERNAL / DATA_DIR / file)['frames']
    n = len(frames)
    rows, maps = [], {}
    for task in tasks(clip, n):
        missing = task['missing']
        idx = np.flatnonzero(missing)
        lengths, positions = run_lengths(missing)
        damaged = frames.copy()
        damaged[missing] = 0
        mask = np.zeros(frames.shape[:3], bool)
        with_dists = split in ('test', 'xiph', 'xiph-c') and task['process'] == 'contiguous' and int(task['task'][3:]) in DISTS_GAPS
        for method in METHODS:
            began = time.perf_counter()
            out, _, _ = restore(damaged, missing, mask, method)
            seconds = (time.perf_counter() - began) / len(idx)
            ref, est = frames[idx], out[idx]
            err = (ref.astype(float) - est.astype(float)) ** 2
            lp = perceptual.distance('lpips', ref, est) if perceptual.available('lpips') else [None] * len(idx)
            if with_dists and perceptual.available('dists'):
                small = lambda a: np.stack([cv2.resize(x, (320, 180), interpolation=cv2.INTER_AREA) for x in a])
                ds = perceptual.distance('dists', small(ref), small(est))
            else:
                ds = [None] * len(idx)
            abs_err = np.abs(ref.astype(np.float32) - est.astype(np.float32)).mean(-1)
            maps[f"{task['task']}/{method}"] = np.stack([cv2.resize(e, MAP_SIZE, interpolation=cv2.INTER_AREA) for e in abs_err]).astype(np.float16)
            for k, i in enumerate(idx):
                mse = float(err[k].mean())
                rows.append(dict(clip=clip, split=split, task=task['task'], process=task['process'], frame=int(i),
                                 run_length=int(lengths[i]), t=round(float(positions[i]), 6), method=method,
                                 psnr=float('inf') if mse == 0 else 10 * np.log10(255 ** 2 / mse),
                                 ssim=float(structural_similarity(ref[k], est[k], channel_axis=-1, data_range=255)),
                                 lpips=None if lp[k] is None else float(lp[k]),
                                 dists=None if ds[k] is None else float(ds[k]), seconds_per_frame=seconds))
        maps[f"{task['task']}/missing"] = missing
    MAPS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MAPS / f'{clip}.npz', **maps)
    print(clip, split, len(rows), 'rows', flush=True)
    return rows


FIELDS = ['clip', 'split', 'task', 'process', 'frame', 'run_length', 't', 'method', 'psnr', 'ssim', 'lpips', 'dists', 'seconds_per_frame']


def run(workers: int, limit: int | None) -> None:
    if DATASET in ('xiph', 'xiph-c'):
        ledger = json.loads((REPO / 'research/data/ledger.json').read_text())
        src = next(s for s in ledger['sources'] if s['name'] == DATASET)
        excluded = set()
        if DATASET == 'xiph-c':   # protocol v4: v3 scene-cut rule applies
            excluded = set(json.loads((REPO / 'research/results/lacunanet/xiph-c-cuts.json').read_text())['excluded'])
        jobs = [(c['id'], DATASET, c['file']) for c in src['sequences'] if c['id'] not in excluded]
    else:
        jobs = [(c['id'], split, c['file']) for split in RUN_SPLITS for c in SPLITS[split]]
    if limit:
        jobs = jobs[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(clip_worker, jobs))
    rows = [r for chunk in results for r in chunk]
    with (OUT / 'rows.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'run.json').write_text(json.dumps(dict(clips=len(jobs), rows=len(rows), workers=workers,
                                                  wall_seconds=time.perf_counter() - began, methods=METHODS, gaps=GAPS,
                                                  loss_rate=RATE, splits='research/protocols/splits-v1.json'), indent=2))
    print(f'{len(rows)} rows in {time.perf_counter() - began:.0f}s')


def load_rows() -> list[dict]:
    with (OUT / 'rows.csv').open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('psnr', 'ssim', 'lpips', 'dists', 't', 'seconds_per_frame'):
            r[k] = float(r[k]) if r[k] not in ('', 'None') else None
        r['run_length'] = int(r['run_length'])
    return rows


def analyse() -> None:
    from backend.stats import bootstrap_mean, holm, paired, scene_means
    rows = load_rows()
    capped = lambda v: min(v, 60.0)  # identical frames give inf PSNR; cap before averaging
    report, lines = dict(gate_G1=None, by_gap={}, by_process={}), []

    def per_scene(sub, method, metric):
        vals = [dict(sequence=r['clip'], v=capped(r[metric]) if metric == 'psnr' else r[metric]) for r in sub if r['method'] == method and r[metric] is not None]
        return scene_means(vals, 'v', 'sequence') if vals else {}

    lines += ['# LacunaBench v1: results (task T3, SNU-FILM)', '',
              'The scene (clip) is the unit: frames are averaged within each clip, and 95% CIs bootstrap over clips. Dev = 20 YouTube clips, test = 11 GoPro clips (frozen in `research/protocols/splits-v1.json`). PSNR is capped at 60 dB before averaging. DISTS is scheduled per amendment 1.', '']
    for split in ('dev', 'test'):
        lines += [f'## {split}: affected-frame PSNR (dB) by gap length', '', '| Gap | ' + ' | '.join(METHODS) + ' |', '|---|' + '---:|' * len(METHODS)]
        for g in GAPS:
            sub = [r for r in rows if r['split'] == split and r['task'] == f'gap{g}']
            cells = []
            for m in METHODS:
                ci = bootstrap_mean(list(per_scene(sub, m, 'psnr').values()))
                cells.append(f'{ci.mean:.2f} [{ci.low:.2f}, {ci.high:.2f}]')
                report['by_gap'].setdefault(split, {}).setdefault(str(g), {})[m] = ci.to_dict()
            lines.append(f'| {g} | ' + ' | '.join(cells) + ' |')
        lines += ['', f'## {split}: LPIPS (lower is better) by gap length', '', '| Gap | ' + ' | '.join(METHODS) + ' |', '|---|' + '---:|' * len(METHODS)]
        for g in GAPS:
            sub = [r for r in rows if r['split'] == split and r['task'] == f'gap{g}']
            cells = []
            for m in METHODS:
                v = per_scene(sub, m, 'lpips')
                cells.append('—' if not v else f'{np.mean(list(v.values())):.3f}')
            lines.append(f'| {g} | ' + ' | '.join(cells) + ' |')
        lines += ['', f'## {split}: loss processes at 25% loss (PSNR dB / LPIPS)', '', '| Process | ' + ' | '.join(METHODS) + ' |', '|---|' + '---:|' * len(METHODS)]
        for proc in ('bernoulli', 'gilbert_elliott', 'gop_chain'):
            sub = [r for r in rows if r['split'] == split and r['task'] == proc]
            cells = []
            for m in METHODS:
                p, l = per_scene(sub, m, 'psnr'), per_scene(sub, m, 'lpips')
                cells.append(f"{np.mean(list(p.values())):.2f} / {np.mean(list(l.values())) if l else float('nan'):.3f}")
                report['by_process'].setdefault(split, {}).setdefault(proc, {})[m] = dict(psnr=float(np.mean(list(p.values()))), lpips=float(np.mean(list(l.values()))) if l else None)
            lines.append(f'| {proc} | ' + ' | '.join(cells) + ' |')
        lines.append('')

    # G1 (pre-registered): quadratic vs two-anchor flow, test split, gaps >= 4, paired over clips.
    sub = [r for r in rows if r['split'] == 'test' and r['process'] == 'contiguous' and int(r['task'][3:]) >= 4]
    quad, flow = per_scene(sub, 'flow_quadratic', 'psnr'), per_scene(sub, 'flow', 'psnr')
    g1 = paired(quad, flow, ('flow_quadratic', 'flow'))
    passed = g1.delta.low > 0
    report['gate_G1'] = dict(**g1.to_dict(), passed=passed, rule='paired dPSNR > 0 with 95% CI excluding 0, test split, gaps >= 4')
    lines += ['## Gate G1 (pre-registered): multi-anchor vs two-anchor flow', '',
              f'Test split, gaps ≥ 4, paired over {g1.delta.n_scenes} clips: ΔPSNR = **{g1.delta.mean:+.3f} dB**, 95% CI [{g1.delta.low:+.3f}, {g1.delta.high:+.3f}], {g1.wins}/{g1.delta.n_scenes} clips improved, Wilcoxon p = {g1.wilcoxon_p if g1.wilcoxon_p is None else round(g1.wilcoxon_p, 4)}.',
              '', f"**G1 {'PASSED' if passed else 'FAILED'}.** " + ('Multi-anchor reconstruction is retained.' if passed else 'Reported as a negative result; two-anchor flow stays the classical default.'), '']

    # Secondary: paired comparisons vs RIFE on the test split, all gaps.
    sub = [r for r in rows if r['split'] == 'test' and r['process'] == 'contiguous']
    rife = per_scene(sub, 'rife', 'psnr')
    comps = holm([paired(per_scene(sub, m, 'psnr'), rife, (m, 'rife')) for m in METHODS if m != 'rife'])
    report['test_vs_rife'] = [c.to_dict() for c in comps]
    lines += ['## Test split: each method vs RIFE (all gaps, paired over clips, Holm-corrected)', '', '| Method | ΔPSNR vs RIFE | 95% CI | Clips better | Holm p |', '|---|---:|---|---:|---:|']
    for c in comps:
        lines.append(f'| {c.method} | {c.delta.mean:+.2f} | [{c.delta.low:+.2f}, {c.delta.high:+.2f}] | {c.wins}/{c.delta.n_scenes} | {"—" if c.holm_p is None else f"{c.holm_p:.4f}"} |')

    dists_rows = [r for r in rows if r['dists'] is not None]
    if dists_rows:
        lines += ['', '## Test split: DISTS at 320×180, gaps 4 and 16 (lower is better)', '', '| Gap | ' + ' | '.join(METHODS) + ' |', '|---|' + '---:|' * len(METHODS)]
        for g in DISTS_GAPS:
            sub = [r for r in dists_rows if r['task'] == f'gap{g}']
            lines.append(f'| {g} | ' + ' | '.join(f"{np.mean(list(per_scene(sub, m, 'dists').values())):.3f}" for m in METHODS) + ' |')
    run_info = json.loads((OUT / 'run.json').read_text())
    lines += ['', f"Run: {run_info['clips']} clips, {run_info['rows']} frame-method rows, {run_info['wall_seconds']/60:.1f} min wall on {run_info['workers']} CPU workers. Error maps (80×45, float16) are stored outside git for C2/C3."]
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2))
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    _figure(rows)
    print('\n'.join(lines))


def _figure(rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from backend.stats import bootstrap_mean, scene_means
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = dict(hold='#9aa3ad', linear='#6b7a89', flow='#4b82ac', flow_quadratic='#2f5d8a', rife='#c87553')
    for ax, metric, label in [(axes[0], 'psnr', 'Affected PSNR (dB) ↑'), (axes[1], 'lpips', 'LPIPS ↓')]:
        for m in METHODS:
            means, lows, highs = [], [], []
            for g in GAPS:
                sub = [dict(sequence=r['clip'], v=min(r[metric], 60.) if metric == 'psnr' else r[metric]) for r in rows
                       if r['split'] == 'test' and r['task'] == f'gap{g}' and r['method'] == m and r[metric] is not None]
                ci = bootstrap_mean(list(scene_means(sub, 'v').values()))
                means.append(ci.mean); lows.append(ci.low); highs.append(ci.high)
            ax.plot(GAPS, means, marker='o', label=m, color=colors[m])
            ax.fill_between(GAPS, lows, highs, color=colors[m], alpha=.12)
        ax.set_xscale('log', base=2); ax.set_xticks(GAPS); ax.set_xticklabels(GAPS)
        ax.set(xlabel='Missing-run length (frames at 30 fps)', ylabel=label); ax.grid(alpha=.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('LacunaBench v1 test split (11 GoPro clips): quality vs gap length, 95% CI over clips', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'gap-curves.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'analyse'])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--clips', type=int, default=None, help='limit for smoke tests')
    args = parser.parse_args()
    run(args.workers, args.clips) if args.command == 'run' else analyse()
