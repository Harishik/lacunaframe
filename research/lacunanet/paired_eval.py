"""Paired, clip-clustered comparison of trained v3 checkpoints against the untrained prior on ALL dev clips.

Per missing frame: PSNR(model) - PSNR(prior-only, i.e. every ReZero gate zeroed: exactly the Lucas-Kanade
pipeline). 95% CI by bootstrap over clips (clips are the independent units). Dev only, no test data.

    ../.venv-train/bin/python research/lacunanet/paired_eval.py name=state.pt [name=state.pt ...]
"""
import copy, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np, torch
from research.lacunanet.model import make_cond
from research.lacunanet.model_v2 import LacunaNetV2
from research.lacunabench.recoverability import brackets
from research.scripts.lacunabench_v1 import tasks

EXT = REPO.parent / 'lacuna-data'
GAPS = ('gap1', 'gap2', 'gap4', 'gap8')


def prior_only(m):
    m = copy.deepcopy(m)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith('gain'):
                p.zero_()
    return m


@torch.no_grad()
def psnrs(models):
    splits = json.loads((REPO / 'research/protocols/splits-v1.json').read_text())
    T = lambda a: torch.from_numpy(a.astype(np.float32).transpose(2, 0, 1)[None] / 255.)
    rows = []
    for c in splits['dev']:
        fr = np.load(EXT / 'snu-film' / c['file'])['frames'][:, :352]
        for task in tasks(c['id'], len(fr)):
            if task['task'] not in GAPS:
                continue
            for left, right, l2, r2, idx in brackets(task['missing']):
                span, gap = right - left, right - left - 1
                for i in idx:
                    t = (i - left) / span
                    cond = torch.tensor([make_cond(t, gap, -t - (left - l2) / span if l2 is not None else 0,
                                                   1 - t + (r2 - right) / span if r2 is not None else 0, l2 is not None, r2 is not None)])
                    z = torch.zeros_like(T(fr[left]))
                    args = (T(fr[left]), T(fr[right]), T(fr[l2]) if l2 is not None else z, T(fr[r2]) if r2 is not None else z, cond)
                    row = dict(clip=c['id'], task=task['task'])
                    for name, m in models.items():
                        e = m(*args)[0][0].numpy().transpose(1, 2, 0) * 255
                        row[name] = 10 * np.log10(255 ** 2 / max(np.mean((np.clip(e, 0, 255).round() - fr[i]) ** 2), 1e-9))
                    lin = (1 - t) * fr[left].astype(float) + t * fr[right]
                    row['linear'] = 10 * np.log10(255 ** 2 / max(np.mean((lin.round() - fr[i]) ** 2), 1e-9))
                    rows.append(row)
    return rows


def clustered_ci(rows, key, ref, reps=2000, seed=0):
    clips = sorted({r['clip'] for r in rows})
    per = {c: np.array([r[key] - r[ref] for r in rows if r['clip'] == c]) for c in clips}
    rng = np.random.default_rng(seed)
    means = [np.concatenate([per[c] for c in rng.choice(clips, len(clips))]).mean() for _ in range(reps)]
    d = np.concatenate(list(per.values()))
    return dict(mean=round(float(d.mean()), 3), lo=round(float(np.percentile(means, 2.5)), 3), hi=round(float(np.percentile(means, 97.5)), 3), n=int(len(d)), clips=len(clips))


def main():
    torch.set_num_threads(6)
    models = {}
    for spec in sys.argv[1:]:
        name, path = spec.split('=', 1)
        m = LacunaNetV2(24, prior='lk'); m.load_state_dict(torch.load(path, map_location='cpu')['model']); m.eval()
        models[name] = m
        if 'prior' not in models:
            models['prior'] = prior_only(m)
    rows = psnrs(models)
    out = {}
    for g in GAPS + ('all',):
        sub = [r for r in rows if g == 'all' or r['task'] == g]
        out[g] = {k: clustered_ci(sub, k, 'prior') for k in models if k != 'prior'} | dict(linear_vs_prior=clustered_ci(sub, 'linear', 'prior'))
        print(g, json.dumps(out[g]), flush=True)
    (REPO / 'research/results/lacunanet/paired-eval-v3.json').write_text(json.dumps(dict(summary=out, rows=rows), indent=1))


if __name__ == '__main__':
    main()
