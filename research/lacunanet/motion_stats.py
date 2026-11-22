"""Train/dev motion mismatch: distribution of inner-anchor displacement (full-res px, Lucas-Kanade at 1/4)
for GOPRO training samples (as the sampler draws them) vs dev missing-frame brackets. Dev only, no test data.

    ../.venv-train/bin/python research/lacunanet/motion_stats.py [stride ...]
"""
import json, random, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch
from research.lacunanet.classical_flow import lk_flow
from research.lacunanet.data import GoproSource
from research.lacunabench.recoverability import brackets
from research.scripts.lacunabench_v1 import tasks

EXT = REPO.parent / 'lacuna-data'
T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32).transpose(2, 0, 1)[None] / 255.)


def mag(a, b):
    with torch.no_grad():
        return float(lk_flow(T(a), T(b), out_scale=4).norm(dim=1).mean() * 4)


BINS = [0, 2, 4, 8, 16, 32, 64, 1e9]


def hist(v):
    h, _ = np.histogram(np.array(v), bins=BINS); return [round(float(x), 4) for x in h / h.sum()]


def pct(v):
    v = np.array(v); return dict(n=len(v), median=round(float(np.median(v)), 2), p75=round(float(np.percentile(v, 75)), 2), p90=round(float(np.percentile(v, 90)), 2))


def main():
    torch.set_num_threads(2)
    splits = json.loads((REPO / 'research/protocols/splits-v1.json').read_text())
    dev = []
    for c in splits['dev']:
        fr = np.load(EXT / 'snu-film' / c['file'])['frames'][:, :352]
        for task in tasks(c['id'], len(fr)):
            if task['task'] in ('gap1', 'gap2', 'gap4', 'gap8'):
                for left, right, *_ in brackets(task['missing']):
                    dev.append(mag(fr[left], fr[right]))
    src = GoproSource(EXT / 'train' / 'gopro'); rng = random.Random(0)
    train = []
    for _ in range(300):
        seq = rng.choice(src.scenes); g = rng.randint(1, min(16, len(seq) - 3)); left = rng.randint(0, len(seq) - g - 2)
        train.append(mag(seq[left], seq[left + g + 1]))
    out = dict(bins=BINS[:-1] + ['inf'], dev_brackets=pct(dev) | dict(hist=hist(dev)), gopro_sampler=pct(train) | dict(hist=hist(train)))
    (REPO / 'research/results/lacunanet/motion-stats.json').write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
