"""Protocol v4 G10: full official midpoint benchmarks, streamed so nothing large is stored.

    .venv/bin/python research/scripts/standard_benchmarks.py vimeo --methods linear,flow,rife,...
    .venv/bin/python research/scripts/standard_benchmarks.py snu --methods ...
    .venv/bin/python research/scripts/standard_benchmarks.py report

Vimeo90K: all 3,782 test triplets (4 parquet shards, each downloaded, scored and deleted).
SNU-FILM: all 4 x 310 official triplets at native resolution, frames fetched per triplet.
Fixed-shape baseline graphs are exported on demand per padded shape with gate P1
(research/metrics/export_shape.py, run in the training environment). Rows append to CSV,
so methods can be added later (e.g. LacunaNet after training) without rerunning others.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import cv2
import numpy as np

EXTERNAL = Path(os.environ.get('LACUNA_EXTERNAL', REPO.parent / 'lacuna-data')).resolve()
OUT = REPO / 'research/results/standard-benchmarks'
TRAIN_PY = REPO.parent / '.venv-train/bin/python'
HF = 'https://huggingface.co/datasets'
VIMEO = HF + '/paschalc24/vimeo-90k-test-full/resolve/main/data/test-0000{}-of-00004.parquet'
SNU = HF + '/xypeng9903/SNU_FILM/resolve/main/'
FIELDS = ['benchmark', 'level', 'item', 'method', 'psnr', 'ssim', 'lpips']


def _interp(method: str, a: np.ndarray, b: np.ndarray, pairs_for_export=None) -> np.ndarray:
    from backend import external
    if method == 'linear':
        return ((a.astype(np.float64) + b) / 2).round().astype(np.uint8)
    if method == 'flow':
        from backend.core import motion
        return motion(a, b, .5)[0]
    if method == 'rife':
        from backend.learned import interpolate
        return interpolate(a, b, .5)
    if method == 'lacunanet':
        return _lacunanet_mid(a, b)
    try:
        return external.interpolate(method, a, b, .5)
    except RuntimeError as exc:
        if 'No ' not in str(exc) or pairs_for_export is None:
            raise
        _export(method, pairs_for_export)
        external.manifest.cache_clear()
        return external.interpolate(method, a, b, .5)


def _lacunanet_mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint with LacunaNet: two observed anchors at times 0 and 2, target at 1, no outer anchors."""
    from backend import lacunanet
    frames = np.stack([a, a, b])           # index 1 is a placeholder that the model never reads
    return lacunanet.reconstruct(frames, 0, 2, None, None, 1)[0]


def _export(method: str, pairs: np.ndarray) -> None:
    with tempfile.TemporaryDirectory() as d:
        np.savez(Path(d) / 'pairs.npz', pairs=pairs)
        r = subprocess.run([str(TRAIN_PY), str(REPO / 'research/metrics/export_shape.py'), method, str(Path(d) / 'pairs.npz')],
                           capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'export failed for {method}: {r.stderr[-800:]}')
    info = json.loads(r.stdout.strip().splitlines()[-1])[method]
    if not info['p1_passed']:
        raise RuntimeError(f'P1 failed for {method}: {info}')
    print('exported', method, info['file'], 'P1', info['p1_max_abs_error'], flush=True)


def _score(ref: np.ndarray, est: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import structural_similarity
    mse = float(np.mean((ref.astype(np.float64) - est.astype(np.float64)) ** 2))
    return (60.0 if mse == 0 else min(60.0, 10 * np.log10(255 ** 2 / mse))), float(structural_similarity(ref, est, channel_axis=-1, data_range=255))


def _done(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r['level'], r['item'], r['method']) for r in csv.DictReader(f)}


def _append(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def _run_items(bench: str, level: str, items, methods: list[str], path: Path) -> None:
    """items: iterable of (item_id, im0, im1, im2). LPIPS is batched per item across methods."""
    from backend import perceptual
    done = _done(path)
    anchors_by_shape = {}
    for item, a, gt, b in items:
        todo = [m for m in methods if (level, item, m) not in done]
        if not todo:
            continue
        anchors_by_shape.setdefault(a.shape, []).append(np.stack([a, b]))
        pairs = np.stack(anchors_by_shape[a.shape][:8])
        outs = {m: _interp(m, a, b, pairs) for m in todo}
        lp = perceptual.distance('lpips', np.stack([gt] * len(todo)), np.stack([outs[m] for m in todo]))
        rows = []
        for k, m in enumerate(todo):
            p, s = _score(gt, outs[m])
            rows.append(dict(benchmark=bench, level=level, item=item, method=m, psnr=p, ssim=s, lpips=float(lp[k])))
        _append(path, rows)


def vimeo(methods: list[str]) -> None:
    import pyarrow.parquet as pq
    from research.data.fetch import decode_png
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'vimeo-rows.csv'
    for shard in range(4):
        tmp = EXTERNAL / f'vimeo-test-{shard}.parquet.part'
        urllib.request.urlretrieve(VIMEO.format(shard), tmp)
        table = pq.read_table(tmp, columns=['im1', 'im2', 'im3', 'sequence_id'])
        tmp.unlink()

        def items():
            for i in range(table.num_rows):
                ims = [decode_png(table.column(c)[i].as_py()['bytes']) for c in ('im1', 'im2', 'im3')]
                yield f"{shard}:{i}:{table.column('sequence_id')[i].as_py()}", ims[0], ims[1], ims[2]
        _run_items('vimeo90k', 'test', items(), methods, path)
        print('vimeo shard', shard, 'done', flush=True)


def snu(methods: list[str]) -> None:
    from research.data.fetch import decode_png
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'snu-film-rows.csv'
    for level in ('easy', 'medium', 'hard', 'extreme'):
        with urllib.request.urlopen(SNU + f'test-{level}.txt') as r:
            lines = [l.split() for l in r.read().decode().splitlines() if l.strip()]

        def items():
            for paths in lines:
                ims = []
                for p in paths:
                    with urllib.request.urlopen(SNU + p.replace('data/SNU-FILM/', ''), timeout=120) as r:
                        ims.append(decode_png(r.read()))
                yield paths[1].replace('data/SNU-FILM/test/', ''), ims[0], ims[1], ims[2]
        _run_items('snu-film', level, items(), methods, path)
        print('snu-film', level, 'done', flush=True)


def report() -> None:
    from backend.stats import bootstrap_mean
    lines = ['# Standard midpoint benchmarks (full official test sets, CPU)', '',
             'Per-image RGB PSNR/SSIM at native resolution and LPIPS-Alex, averaged over all items (the official convention). Every method ran on the same CPU with identical inputs. Latency is measured separately on an idle machine (`latency.json`).', '']
    latency = json.loads((OUT / 'latency.json').read_text()) if (OUT / 'latency.json').exists() else {}
    for name, file in (('Vimeo90K test (3,782)', 'vimeo-rows.csv'), ('SNU-FILM (4 × 310)', 'snu-film-rows.csv')):
        p = OUT / file
        if not p.exists():
            continue
        with p.open() as f:
            rows = list(csv.DictReader(f))
        levels = sorted({r['level'] for r in rows}, key=['test', 'easy', 'medium', 'hard', 'extreme'].index)
        methods = sorted({r['method'] for r in rows})
        lines += [f'## {name}', '', '| Method | ' + ' | '.join(f'{l} PSNR / SSIM / LPIPS (n)' for l in levels) + ' |', '|---|' + '---:|' * len(levels)]
        for m in methods:
            cells = []
            for l in levels:
                sub = [r for r in rows if r['method'] == m and r['level'] == l]
                if not sub:
                    cells.append('—'); continue
                cells.append(f"{np.mean([float(r['psnr']) for r in sub]):.2f} / {np.mean([float(r['ssim']) for r in sub]):.4f} / {np.mean([float(r['lpips']) for r in sub]):.4f} ({len(sub)})")
            lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['vimeo', 'snu', 'report'])
    ap.add_argument('--methods', default='linear,flow,rife,rife_paper,amt_s,ifrnet_s,ifrnet,amt_s_gopro,ifrnet_gopro')
    a = ap.parse_args()
    {'vimeo': lambda: vimeo(a.methods.split(',')), 'snu': lambda: snu(a.methods.split(',')), 'report': report}[a.command]()
