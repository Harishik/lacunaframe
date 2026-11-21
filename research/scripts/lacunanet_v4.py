"""Protocol v4 evaluation of LacunaNet.

    .venv/bin/python research/scripts/lacunanet_v4.py cuts              # v3 cut rule on Xiph-C (exclusions)
    .venv/bin/python research/scripts/lacunanet_v4.py select            # dev-only checkpoint selection
    .venv/bin/python research/scripts/lacunanet_v4.py selfcheck <set>   # G11 data: self-error vs actual error (dev, xiph-c)
    .venv/bin/python research/scripts/lacunanet_v4.py judge             # G8, G9, G11 on Xiph-C
    .venv/bin/python research/scripts/lacunanet_v4.py latency           # idle-machine CPU latency, all methods
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import cv2
import numpy as np

EXTERNAL = Path(os.environ.get('LACUNA_EXTERNAL', REPO.parent / 'lacuna-data')).resolve()
MODELS = EXTERNAL / 'models' / 'lacunanet'
OUT = REPO / 'research/results/lacunanet'
TRAIN_PY = REPO.parent / '.venv-train/bin/python'
SPLITS = json.loads((REPO / 'research/protocols/splits-v1.json').read_text())
TAU, ALPHA = 20.0, 0.10
TASK_NAMES = ('gap1', 'gap2', 'gap4', 'gap8', 'gap16', 'gap32', 'bernoulli', 'gilbert_elliott', 'gop_chain')


def _ledger_clips(name: str) -> list[dict]:
    led = json.loads((REPO / 'research/data/ledger.json').read_text())
    return next(s for s in led['sources'] if s['name'] == name)['sequences']


def xiph_c_clips() -> list[tuple[str, str]]:
    excluded = set(json.loads((OUT / 'xiph-c-cuts.json').read_text())['excluded'])
    return [(c['id'], c['file']) for c in _ledger_clips('xiph-c') if c['id'] not in excluded]


def cuts() -> None:
    from research.scripts.fusion_v3 import _has_cut
    report = {}
    for c in _ledger_clips('xiph-c'):
        cut, d = _has_cut(np.load(EXTERNAL / 'xiph-c' / c['file'])['frames'])
        report[c['id']] = dict(cut=cut, max_diff=max(d), median_diff=float(np.median(d)))
    excluded = sorted(k for k, v in report.items() if v['cut'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'xiph-c-cuts.json').write_text(json.dumps(dict(rule='v3 amendment 1 cut rule', clips=report, excluded=excluded,
                                                          remaining=len(report) - len(excluded)), indent=2))
    print('excluded:', excluded, 'remaining:', len(report) - len(excluded))


def _frames_for(dataset: str, file: str) -> np.ndarray:
    return np.load(EXTERNAL / {'dev': 'snu-film', 'xiph-c': 'xiph-c'}[dataset] / file)['frames']


def _tasks(clip: str, n: int):
    from research.scripts.lacunabench_v1 import tasks
    return tasks(clip, n)


def _items(dataset: str, every: int = 1):
    """Yield (clip, task, frames, left, right, left2, right2, index, run_length) for missing frames."""
    from research.lacunabench.recoverability import brackets
    clips = [(c['id'], c['file']) for c in SPLITS['dev']] if dataset == 'dev' else xiph_c_clips()
    for clip, file in clips:
        frames = _frames_for(dataset, file)
        for task in _tasks(clip, len(frames)):
            k = 0
            for left, right, l2, r2, idx in brackets(task['missing']):
                for i in idx:
                    if k % every == 0:
                        yield clip, task['task'], frames, left, right, l2, r2, i, len(idx)
                    k += 1


def _export(ckpt: Path, onnx: Path) -> None:
    r = subprocess.run([str(TRAIN_PY), str(REPO / 'research/lacunanet/export.py'), str(ckpt), str(onnx)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-600:])


def select() -> None:
    """Dev-only checkpoint choice: scene-mean PSNR - 20 * LPIPS on every 4th dev missing frame (protocol v4 §4)."""
    from backend import perceptual
    from backend.stats import scene_means
    import backend.lacunanet as ln
    ckpts = sorted(MODELS.glob('ckpt-*.pt'))
    if not ckpts:
        raise SystemExit('no checkpoints')
    results = {}
    for ck in ckpts:
        onnx = MODELS / f'{ck.stem}.onnx'
        if not onnx.exists():
            _export(ck, onnx)
        os.environ['LACUNANET_ONNX'] = str(onnx)
        ln._session.cache_clear()
        rows = []
        for clip, task, frames, left, right, l2, r2, i, _ in _items('dev', every=4):
            out, _ = ln.reconstruct(frames, left, right, l2, r2, i)
            mse = float(np.mean((out.astype(float) - frames[i].astype(float)) ** 2))
            rows.append(dict(sequence=clip, psnr=min(60., 10 * np.log10(255 ** 2 / max(mse, 1e-12))),
                             lpips=float(perceptual.distance('lpips', frames[i][None], out[None])[0])))
        p = float(np.mean(list(scene_means(rows, 'psnr').values())))
        l = float(np.mean(list(scene_means(rows, 'lpips').values())))
        results[ck.name] = dict(psnr=p, lpips=l, score=p - 20 * l, frames=len(rows))
        print(ck.name, results[ck.name], flush=True)
    best = max(results, key=lambda k: results[k]['score'])
    chosen = MODELS / 'lacunanet.onnx'
    src = MODELS / f'{Path(best).stem}.onnx'
    chosen.write_bytes(src.read_bytes())
    chosen.with_suffix('.json').write_text(json.dumps(dict(checkpoint=best, sha256=hashlib.sha256(chosen.read_bytes()).hexdigest()), indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'checkpoint-selection.json').write_text(json.dumps(dict(rule='dev SNU YouTube T3, every 4th missing frame, scene-mean PSNR - 20*LPIPS', chosen=best, results=results), indent=2))
    print('chosen', best)


def selfcheck(dataset: str) -> None:
    """Per missing frame: LacunaNet output metrics, pooled predicted-error map and pooled actual-error map (80x45)."""
    from research.lacunabench.fusion import pooled_error
    import backend.lacunanet as ln
    store, rows = {}, []
    for clip, task, frames, left, right, l2, r2, i, run in _items(dataset):
        out, scale = ln.reconstruct(frames, left, right, l2, r2, i)
        actual = pooled_error(frames[i], out)
        pred = cv2.resize(scale.astype(np.float32), (80, 45), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((out.astype(float) - frames[i].astype(float)) ** 2))
        rows.append(dict(clip=clip, task=task, frame=int(i), run=int(run), psnr=min(60., 10 * np.log10(255 ** 2 / max(mse, 1e-12))),
                         pred_mean=float(pred.mean())))
        store.setdefault(clip, {}).setdefault(task, []).append((pred, actual, run))
    folder = EXTERNAL / 'results' / 'lacunanet-selfcheck' / dataset
    folder.mkdir(parents=True, exist_ok=True)
    for clip, tasks in store.items():
        np.savez_compressed(folder / f'{clip}.npz', **{f'{t}/pred': np.stack([p for p, _, _ in v]).astype(np.float16) for t, v in tasks.items()},
                            **{f'{t}/actual': np.stack([a for _, a, _ in v]).astype(np.float16) for t, v in tasks.items()},
                            **{f'{t}/run': np.array([r for _, _, r in v]) for t, v in tasks.items()})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f'selfcheck-{dataset}.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(dataset, len(rows), 'frames')


def latency() -> None:
    """Idle-machine median CPU latency (2 threads, warm, 20 runs) per method and resolution."""
    from backend import external
    from backend.core import motion
    from backend.learned import interpolate
    import backend.lacunanet as ln
    rng = np.random.default_rng(0)
    res = {}
    for h, w in ((360, 640), (256, 448), (720, 1280)):
        a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        b = np.roll(a, 3, axis=1)
        fns = {'linear': lambda: ((a.astype(float) + b) / 2).astype(np.uint8), 'flow': lambda: motion(a, b, .5),
               'rife': lambda: interpolate(a, b, .5)}
        for name in ('amt_s', 'ifrnet_s', 'ifrnet', 'amt_s_gopro', 'ifrnet_gopro', 'rife_paper'):
            fns[name] = (lambda n: lambda: external.interpolate(n, a, b, .5))(name)
        if ln.available():
            fr = np.stack([a, a, b])
            fns['lacunanet'] = lambda: ln.reconstruct(fr, 0, 2, None, None, 1)
        for name, fn in fns.items():
            try:
                for _ in range(3):
                    fn()
                ts = []
                for _ in range(20):
                    t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
                res.setdefault(f'{w}x{h}', {})[name] = float(np.median(ts))
            except RuntimeError as exc:
                res.setdefault(f'{w}x{h}', {})[name] = None
                print('skip', name, w, h, str(exc)[:80])
            print(w, h, name, res[f'{w}x{h}'][name], flush=True)
    target = REPO / 'research/results/standard-benchmarks/latency.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(threads=2, note='median of 20 warm runs on an otherwise idle machine; None = no export at that shape', seconds=res), indent=2))


if __name__ == '__main__':
    cmd = sys.argv[1]
    {'cuts': cuts, 'select': select, 'selfcheck': lambda: selfcheck(sys.argv[2]), 'latency': latency}[cmd]()
