"""T1 parity check: midpoint interpolation on 256 Vimeo90K test triplets (native 448x256).

Checks that our RIFE pipeline (pinned Practical-RIFE 4.9 ONNX, CPU) lands in the
range the literature reports for RIFE-family models on Vimeo90K, using the
common convention (per-image RGB PSNR, averaged). It is a sanity check of the
pipeline, not a reproduction of any paper's table: 4.9 is a Practical-RIFE
release, not the ECCV 2022 checkpoint, and this is a 256-triplet subset.
"""
from pathlib import Path
import json
import os
import sys
import time
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np
from skimage.metrics import structural_similarity
from backend.core import motion
from backend.learned import interpolate

EXTERNAL = Path(os.environ.get('LACUNA_EXTERNAL', REPO.parent / 'lacuna-data')).resolve()
OUT = REPO / 'research/results/vimeo-parity'
REFERENCE = {'RIFE (ECCV 2022, paper Table)': 35.62, 'IFRNet-S (CVPR 2022, paper)': 35.59, 'AMT-S (CVPR 2023, paper)': 35.97}


def score(triplets: np.ndarray) -> dict:
    methods = {'linear': lambda a, b: ((a.astype(float) + b) / 2).round().astype(np.uint8),
               'flow': lambda a, b: motion(a, b, .5)[0],
               'rife': lambda a, b: interpolate(a, b, .5)}
    results = {}
    for name, fn in methods.items():
        psnrs, ssims, began = [], [], time.perf_counter()
        for t in triplets:
            est = fn(t[0], t[2])
            mse = np.mean((est.astype(float) - t[1].astype(float)) ** 2)
            psnrs.append(10 * np.log10(255 ** 2 / mse))
            ssims.append(structural_similarity(est, t[1], channel_axis=-1, data_range=255))
        results[name] = dict(psnr=float(np.mean(psnrs)), psnr_std=float(np.std(psnrs)), ssim=float(np.mean(ssims)),
                             seconds_per_frame=(time.perf_counter() - began) / len(triplets))
        print(name, results[name], flush=True)
    return results


SUBSETS = {'shard0-256.npz': 'shard 0 only (first quarter of the test set, ordered by sequence)',
           'balanced-256.npz': '64 evenly spaced triplets from each of the 4 shards'}


def main() -> None:
    out, lines = {}, ['# Vimeo90K T1 parity check (448×256 midpoint)', '',
                      'Per-image RGB PSNR, averaged. Published numbers are for the full 3,782-triplet test set and other checkpoints, so they are context, not a matched comparison. Timings were taken while another benchmark shared the CPU and are indicative only.', '']
    for file, label in SUBSETS.items():
        path = EXTERNAL / 'vimeo-triplets' / file
        if not path.exists():
            continue
        triplets = np.load(path)['triplets']
        results = score(triplets)
        out[file] = dict(label=label, triplets=int(len(triplets)), results=results)
        lines += [f'## {label} ({len(triplets)} triplets)', '', '| Method | PSNR (dB) | SSIM | s / frame (CPU) |', '|---|---:|---:|---:|']
        lines += [f"| {k} | {v['psnr']:.2f} | {v['ssim']:.4f} | {v['seconds_per_frame']:.3f} |" for k, v in results.items()]
        lines.append('')
    lines += ['Published, full test set: ' + '; '.join(f'{k} {v:.2f} dB' for k, v in REFERENCE.items()) + '.']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(dict(subsets=out, published_reference_full_test_set=REFERENCE), indent=2))
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
