"""Protocol v3: recoverability-guided fusion (RGF) and label-free adaptive calibration (LFAC).

    .venv/bin/python research/scripts/fusion_v3.py devpass     # dev candidates, disagreement, error maps
    .venv/bin/python research/scripts/fusion_v3.py fit         # CV temperature, OOF calibration, final models
    .venv/bin/python research/scripts/fusion_v3.py testpass [dataset]   # default xiph-b
    .venv/bin/python research/scripts/fusion_v3.py judge [dataset]

Everything that is fit (error predictors, classifier, T, lambda_dev) uses the v1
dev split only. Test sets are only scored.
"""
from __future__ import annotations
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np

EXTERNAL = Path(os.environ.get('LACUNA_EXTERNAL', REPO.parent / 'lacuna-data')).resolve()
SPLITS = json.loads((REPO / 'research/protocols/splits-v1.json').read_text())
PROTOCOL = json.loads((REPO / 'research/protocols/recoverability-v3.json').read_text())
WORK = EXTERNAL / 'results/fusion-v3'
MODELS = EXTERNAL / 'models/fusion-v3'
OUT = REPO / 'research/results/fusion-v3'
V1_FEATS = EXTERNAL / 'results/recoverability-v1'
TAU, ALPHA, SEED = 20.0, 0.10, 20261120
T_GRID = tuple(PROTOCOL['fusion']['T_grid'])
PX_PER_FRAME = 300
DATASETS = {'xiph-b': ('xiph-b', 'xiph-b'), 'xiph': ('xiph', 'xiph'), 'xiph-c': ('xiph-c', 'xiph-c'), 'snu-test': ('snu-film', None)}


def _tasks(clip: str, n: int):
    from research.scripts.lacunabench_v1 import tasks
    return tasks(clip, n)


def _candidates(frames: np.ndarray, missing: np.ndarray) -> dict[str, np.ndarray]:
    from backend.core import restore
    from research.lacunabench.fusion import CANDIDATES
    mask = np.zeros(frames.shape[:3], bool)
    damaged = frames.copy()
    damaged[missing] = 0
    return {m: restore(damaged, missing, mask, m)[0] for m in CANDIDATES}


def _run_lengths(missing: np.ndarray) -> np.ndarray:
    from research.scripts.recoverability_v1 import _run_lengths
    return _run_lengths(missing)


# --------------------------------------------------------------------------- dev pass

def dev_clip(args: tuple[str, str]) -> str:
    clip, file = args
    os.environ.setdefault('LACUNA_CPU_THREADS', '2')
    from research.lacunabench.fusion import CANDIDATES, disagreement, pooled_error
    frames = np.load(EXTERNAL / 'snu-film' / file)['frames']
    out = {}
    for task in _tasks(clip, len(frames)):
        missing, name = task['missing'], task['task']
        idx = np.flatnonzero(missing)
        cands = _candidates(frames, missing)
        stacks = [np.stack([cands[m][i] for m in CANDIDATES]) for i in idx]
        out[f'{name}/dis'] = np.stack([disagreement(s) for s in stacks]).astype(np.float16)
        out[f'{name}/err'] = np.stack([[pooled_error(frames[i], cands[m][i]) for m in CANDIDATES] for i in idx]).astype(np.float16)
        sub = np.arange(0, len(idx), 4)
        out[f'{name}/sub_k'] = sub
        out[f'{name}/sub_cand'] = np.stack([stacks[k] for k in sub])
        out[f'{name}/missing'] = missing
    (WORK / 'dev').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK / 'dev' / f'{clip}.npz', **out)
    print('devpass', clip, flush=True)
    return clip


def devpass(workers: int = 4) -> None:
    jobs = [(c['id'], c['file']) for c in SPLITS['dev']]
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(dev_clip, jobs))


# --------------------------------------------------------------------------- fit

def _dev_frames() -> list[dict]:
    """One record per dev missing frame: clip, task, run length, rec maps, dis maps, err maps, subset slot."""
    records = []
    for c in SPLITS['dev']:
        clip = c['id']
        rec, fz = np.load(V1_FEATS / f'{clip}.npz'), np.load(WORK / 'dev' / f'{clip}.npz')
        for task in ('gap1', 'gap2', 'gap4', 'gap8', 'gap16', 'gap32', 'bernoulli', 'gilbert_elliott', 'gop_chain'):
            missing = fz[f'{task}/missing']
            idx, runs = np.flatnonzero(missing), _run_lengths(missing)
            sub = {int(k): j for j, k in enumerate(fz[f'{task}/sub_k'])}
            for k, i in enumerate(idx):
                records.append(dict(clip=clip, task=task, frame=int(i), run=int(runs[i]), rec=rec[f'{task}/maps'][k].astype(np.float32),
                                    dis=fz[f'{task}/dis'][k].astype(np.float32), err=fz[f'{task}/err'][k].astype(np.float32),
                                    sub=(task, sub[k]) if k in sub else None, file=c['file']))
    return records


def _train_tables(records: list[dict], rng: np.random.Generator):
    from research.lacunabench.fusion import pixel_features
    X, Y = [], []
    for r in records:
        rows = pixel_features(r['rec'], r['dis'])
        pick = rng.choice(len(rows), PX_PER_FRAME, replace=False)
        X.append(rows[pick])
        Y.append(r['err'].reshape(len(r['err']), -1)[:, pick].T)
    return np.vstack(X), np.vstack(Y)


def _fit_models(X: np.ndarray, Y: np.ndarray):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    kw = dict(max_iter=300, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)
    regs = [HistGradientBoostingRegressor(**kw).fit(X, Y[:, k]) for k in range(Y.shape[1])]
    clf = HistGradientBoostingClassifier(**kw).fit(X, Y[:, 0] > TAU)  # candidate 0 = RIFE 4.9
    return regs, clf


def _predict(regs, clf, rec: np.ndarray, dis: np.ndarray):
    from research.lacunabench.fusion import pixel_features
    rows = pixel_features(rec, dis)
    pred = np.stack([np.maximum(m.predict(rows), 0).reshape(45, 80) for m in regs])
    prob = clf.predict_proba(rows)[:, 1].reshape(45, 80)
    return pred, prob


def fit() -> None:
    import joblib
    from backend import perceptual
    from backend.stats import scene_means
    from research.lacunabench.conformal import Unit, calibrate
    from research.lacunabench.fusion import CANDIDATES, fuse
    rng = np.random.default_rng(SEED)
    records = _dev_frames()
    clips = sorted({r['clip'] for r in records})
    fold = {c: i % 5 for i, c in enumerate(clips)}
    subsets = {c: np.load(WORK / 'dev' / f'{c}.npz') for c in clips}
    tune = {T: [] for T in T_GRID}
    rife_rows, oof, refs = [], [], {}
    for f in range(5):
        train = [r for r in records if fold[r['clip']] != f]
        regs, clf = _fit_models(*_train_tables(train, rng))
        for r in (r for r in records if fold[r['clip']] == f):
            pred, prob = _predict(regs, clf, r['rec'], r['dis'])
            oof.append(dict(clip=r['clip'], run=r['run'], prob=prob.reshape(-1), err=r['err'][0].reshape(-1)))
            if r['sub'] is None:
                continue
            task, j = r['sub']
            stack = subsets[r['clip']][f'{task}/sub_cand'][j]
            if r['clip'] not in refs:
                refs.clear()
                refs[r['clip']] = np.load(EXTERNAL / 'snu-film' / r['file'])['frames']
            ref = refs[r['clip']][r['frame']]
            mse = lambda x: float(np.mean((x.astype(float) - ref.astype(float)) ** 2))
            rife_rows.append(dict(sequence=r['clip'], psnr=10 * np.log10(255 ** 2 / max(mse(stack[0]), 1e-12)),
                                  lpips=float(perceptual.distance('lpips', ref[None], stack[0][None])[0])))
            for T in T_GRID:
                out = fuse(stack, pred, T)
                tune[T].append(dict(sequence=r['clip'], psnr=10 * np.log10(255 ** 2 / max(mse(out), 1e-12)),
                                    lpips=float(perceptual.distance('lpips', ref[None], out[None])[0])))
        print('fold', f, 'done', flush=True)
    agg = lambda rows, key: float(np.mean(list(scene_means(rows, key).values())))
    rife_psnr, rife_lpips = agg(rife_rows, 'psnr'), agg(rife_rows, 'lpips')
    table = {T: dict(psnr=agg(v, 'psnr'), lpips=agg(v, 'lpips')) for T, v in tune.items()}
    ok = [T for T in T_GRID if table[T]['lpips'] <= rife_lpips + .005]
    T_star = max(ok, key=lambda T: table[T]['psnr']) if ok else min(T_GRID, key=lambda T: table[T]['lpips'])
    # Out-of-fold classifier scores -> dev CRC lambda per Mondrian stratum (units = clips).
    lam = {}
    for name, sel in [('short', lambda r: r['run'] <= 4), ('long', lambda r: r['run'] > 4)]:
        units = []
        for c in clips:
            rows = [o for o in oof if o['clip'] == c and sel(o)]
            if rows:
                units.append(Unit(np.concatenate([o['prob'] for o in rows]), np.concatenate([o['err'] for o in rows])))
        lam[name] = calibrate(units, ALPHA, TAU)
    regs, clf = _fit_models(*_train_tables(records, rng))
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(regs=regs, clf=clf, candidates=CANDIDATES), MODELS / 'models.joblib')
    config = dict(T_star=T_star, tuning=dict(rife=dict(psnr=rife_psnr, lpips=rife_lpips), by_T={str(k): v for k, v in table.items()}),
                  lambda_dev=lam, frames_tuned=len(rife_rows), dev_frames=len(records), rule=PROTOCOL['fusion']['T_rule'])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'fit.json').write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2))


# --------------------------------------------------------------------------- test pass

def test_clip(args: tuple[str, str, str]) -> list[dict]:
    clip, file, dataset = args
    os.environ.setdefault('LACUNA_CPU_THREADS', '2')
    import joblib
    from skimage.metrics import structural_similarity
    from backend import perceptual
    from research.lacunabench.fusion import CANDIDATES, disagreement, fuse, pooled_error
    from research.lacunabench.recoverability import BracketFeatures, brackets
    T = json.loads((OUT / 'fit.json').read_text())['T_star']
    models = joblib.load(MODELS / 'models.joblib')
    frames = np.load(EXTERNAL / DATASETS[dataset][0] / file)['frames']
    rows, store = [], {}
    for task in _tasks(clip, len(frames)):
        missing, name = task['missing'], task['task']
        idx, runs = np.flatnonzero(missing), _run_lengths(missing)
        rec = {}
        for left, right, l2, r2, members in brackets(missing):
            bf = BracketFeatures(frames, left, right, l2, r2)
            for i in members:
                rec[i] = bf.maps(i)
        cands = _candidates(frames, missing)
        outputs = {m: [] for m in (*CANDIDATES, 'rgf')}
        probs, rife_err = [], []
        for i in idx:
            stack = np.stack([cands[m][i] for m in CANDIDATES])
            pred, prob = _predict(models['regs'], models['clf'], rec[i], disagreement(stack))
            for m, x in zip(CANDIDATES, stack):
                outputs[m].append(x)
            outputs['rgf'].append(fuse(stack, pred, T))
            probs.append(prob.astype(np.float32))
            rife_err.append(pooled_error(frames[i], cands['rife'][i]))
        ref = frames[idx]
        for m, est in outputs.items():
            est = np.stack(est)
            lp = perceptual.distance('lpips', ref, est)
            for k, i in enumerate(idx):
                mse = float(np.mean((ref[k].astype(float) - est[k].astype(float)) ** 2))
                rows.append(dict(clip=clip, task=name, process=task['process'], frame=int(i), run_length=int(runs[i]), method=m,
                                 psnr=float('inf') if mse == 0 else 10 * np.log10(255 ** 2 / mse),
                                 ssim=float(structural_similarity(ref[k], est[k], channel_axis=-1, data_range=255)), lpips=float(lp[k])))
        store[f'{name}/prob'] = np.stack(probs)
        store[f'{name}/rife_err'] = np.stack(rife_err).astype(np.float16)
        store[f'{name}/run'] = runs[idx]
    (WORK / dataset).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(WORK / dataset / f'{clip}.npz', **store)
    print('testpass', dataset, clip, flush=True)
    return rows


def _clips(dataset: str) -> list[tuple[str, str]]:
    if dataset == 'snu-test':
        return [(c['id'], c['file']) for c in SPLITS['test']]
    ledger = json.loads((REPO / 'research/data/ledger.json').read_text())
    src = next(s for s in ledger['sources'] if s['name'] == DATASETS[dataset][1])
    excluded = set()
    if dataset == 'xiph-b':
        excluded = set(json.loads((OUT / 'xiph-b-cuts.json').read_text())['xiph-b_excluded'])
    elif dataset == 'xiph-c':
        excluded = set(json.loads((REPO / 'research/results/lacunanet/xiph-c-cuts.json').read_text())['excluded'])
    return [(c['id'], c['file']) for c in src['sequences'] if c['id'] not in excluded]


def testpass(dataset: str = 'xiph-b', workers: int = 4) -> None:
    jobs = [(c, f, dataset) for c, f in _clips(dataset)]
    with ProcessPoolExecutor(workers) as pool:
        rows = [r for chunk in pool.map(test_clip, jobs) for r in chunk]
    (OUT / dataset).mkdir(parents=True, exist_ok=True)
    with (OUT / dataset / 'rows.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), 'rows')


# --------------------------------------------------------------------------- judge

def judge(dataset: str = 'xiph-b') -> None:
    from backend.stats import bootstrap_mean, holm, paired, scene_means
    from research.lacunabench.conformal import Unit, evaluate as crc_eval
    from research.lacunabench.fusion import CANDIDATES
    fitted = json.loads((OUT / 'fit.json').read_text())
    with (OUT / dataset / 'rows.csv').open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['psnr'], r['lpips'] = min(float(r['psnr']), 60.), float(r['lpips'])
    per = lambda sub, m, key: scene_means([dict(sequence=r['clip'], v=r[key]) for r in sub if r['method'] == m], 'v')
    methods = (*CANDIDATES, 'rgf')
    report, lines = dict(dataset=dataset, T_star=fitted['T_star']), []
    gated = dataset == 'xiph-b'
    lines += [f'# Protocol v3 on {dataset}: recoverability-guided fusion (RGF) and label-free adaptive calibration', '',
              f"{'Untouched test set; gates G5–G6 are judged here.' if gated else 'Secondary (already-seen) set: descriptive only, no gates.'} Everything fitted came from the 20 dev clips. T* = {fitted['T_star']} (chosen on dev by the pre-registered rule). The clip is the unit.", '']
    # G5: all T3 tasks.
    p_rgf, p_rife = per(rows, 'rgf', 'psnr'), per(rows, 'rife', 'psnr')
    l_rgf, l_rife = per(rows, 'rgf', 'lpips'), per(rows, 'rife', 'lpips')
    d = paired(p_rgf, p_rife, ('rgf', 'rife'))
    dl = paired(l_rgf, l_rife, ('rgf', 'rife'))
    g5 = d.delta.low > 0 and dl.delta.mean <= .005
    report['G5'] = dict(psnr=d.to_dict(), lpips=dl.to_dict(), passed=g5)
    lines += ['## G5: RGF vs RIFE 4.9 (all gaps and loss processes)', '',
              f'ΔPSNR = **{d.delta.mean:+.3f} dB** [{d.delta.low:+.3f}, {d.delta.high:+.3f}], {d.wins}/{d.delta.n_scenes} clips better, Wilcoxon p = {d.wilcoxon_p if d.wilcoxon_p is None else round(d.wilcoxon_p, 4)}. ΔLPIPS = {dl.delta.mean:+.4f} [{dl.delta.low:+.4f}, {dl.delta.high:+.4f}].',
              f"**G5 {'PASSED' if g5 else 'FAILED'}**" + ('' if gated else ' (descriptive)') + '.', '']
    # Method table and G5b.
    lines += ['## All methods by gap length (PSNR dB / LPIPS, scene means)', '', '| Gap | ' + ' | '.join(methods) + ' |', '|---|' + '---:|' * len(methods)]
    g5b, worst = True, None
    for g in (1, 2, 4, 8, 16, 32):
        sub = [r for r in rows if r['task'] == f'gap{g}']
        ps = {m: float(np.mean(list(per(sub, m, 'psnr').values()))) for m in methods}
        ls = {m: float(np.mean(list(per(sub, m, 'lpips').values()))) for m in methods}
        best = max(CANDIDATES, key=ps.get)
        margin = ps['rgf'] - ps[best]
        g5b &= margin >= -.1
        worst = margin if worst is None else min(worst, margin)
        report.setdefault('by_gap', {})[g] = dict(psnr=ps, lpips=ls, best_single=best, rgf_minus_best=margin)
        lines.append(f'| {g} | ' + ' | '.join(f"{'**' if m == 'rgf' else ''}{ps[m]:.2f} / {ls[m]:.3f}{'**' if m == 'rgf' else ''}" for m in methods) + ' |')
    for proc in ('bernoulli', 'gilbert_elliott', 'gop_chain'):
        sub = [r for r in rows if r['task'] == proc]
        lines.append(f'| {proc} | ' + ' | '.join(f"{np.mean(list(per(sub, m, 'psnr').values())):.2f} / {np.mean(list(per(sub, m, 'lpips').values())):.3f}" for m in methods) + ' |')
    report['G5b'] = dict(worst_margin_vs_best_single=worst, passed=g5b)
    lines += ['', f"**G5b {'PASSED' if g5b else 'FAILED'}**: worst per-gap margin of RGF over the best single candidate = {worst:+.2f} dB (floor −0.10).", '']
    comps = holm([paired(p_rgf, per(rows, m, 'psnr'), ('rgf', m)) for m in CANDIDATES])
    lines += ['## RGF vs each candidate (all tasks, paired over clips, Holm-corrected)', '', '| vs | ΔPSNR | 95% CI | Clips better | Holm p |', '|---|---:|---|---:|---:|']
    lines += [f'| {c.baseline} | {c.delta.mean:+.2f} | [{c.delta.low:+.2f}, {c.delta.high:+.2f}] | {c.wins}/{c.delta.n_scenes} | {"—" if c.holm_p is None else f"{c.holm_p:.4f}"} |' for c in comps]
    report['vs_candidates'] = [c.to_dict() for c in comps]
    # G6/G7: LFAC on RIFE errors.
    units = {}
    for c, _ in _clips(dataset):
        z = np.load(WORK / dataset / f'{c}.npz')
        for task in ('gap1', 'gap2', 'gap4', 'gap8', 'gap16', 'gap32', 'bernoulli', 'gilbert_elliott', 'gop_chain'):
            prob, err, run = z[f'{task}/prob'], z[f'{task}/rife_err'].astype(np.float32), z[f'{task}/run']
            for stratum, sel in (('short', run <= 4), ('long', run > 4)):
                if sel.any():
                    units.setdefault(stratum, {}).setdefault(task, []).append((c, prob[sel].reshape(-1), err[sel].reshape(-1)))
    lines += ['', '## G6/G7: label-free adaptive calibration (RIFE errors, τ = 20, α = 0.1)', '',
              '| Stratum | λ_dev | λ_target (no labels) | Deployed λ | Coverage (LFAC) | Area (LFAC) | Coverage (dev-only λ) | Area (dev-only λ) | Clips with bad pixels |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    worst_cov, report['G6'] = 1.0, {}
    for stratum, by_task in units.items():
        pooled = np.concatenate([p for t in by_task.values() for _, p, _ in t])
        # Label-free estimate: FNR(lam) ~= sum_{p<lam} p / sum p; take the largest lam with estimate <= alpha.
        order = np.sort(pooled)
        cum = np.cumsum(order) / order.sum()
        k = np.searchsorted(cum, ALPHA, side='right')
        lam_target = float(order[k - 1]) if k > 0 else float(order[0])
        lam_dev = fitted['lambda_dev'][stratum]
        lam = min(lam_dev, lam_target)
        for task, triples in by_task.items():
            us = [Unit(p, e) for _, p, e in triples]
            a, b = crc_eval(us, lam, TAU), crc_eval(us, lam_dev, TAU)
            report['G6'][f'{task}/{stratum}'] = dict(lambda_dev=lam_dev, lambda_target=lam_target, lam=lam, coverage=a['coverage'], area=a['flagged_fraction'],
                                                     coverage_dev_only=b['coverage'], area_dev_only=b['flagged_fraction'], clips_with_bad=a['units_with_bad_pixels'])
            if a['units_with_bad_pixels'] >= 3:
                worst_cov = min(worst_cov, a['coverage'])
            lines.append(f"| {task} / {stratum} | {lam_dev:.3f} | {lam_target:.3f} | {lam:.3f} | {a['coverage']:.3f} | {a['flagged_fraction']:.3f} | {b['coverage']:.3f} | {b['flagged_fraction']:.3f} | {a['units_with_bad_pixels']} |")
    g6 = worst_cov >= 1 - ALPHA - .02
    report['G6_gate'] = dict(worst_coverage=worst_cov, passed=g6)
    lines += ['', f"**G6 {'PASSED' if g6 else 'FAILED'}**" + ('' if gated else ' (descriptive)') + f': worst gated stratum coverage {worst_cov:.3f} (floor 0.88).']
    (OUT / dataset / 'summary.json').write_text(json.dumps(report, indent=2, default=float))
    (OUT / dataset / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# --------------------------------------------------------------------------- scene-cut rule (v3 amendment 1)

def _has_cut(frames: np.ndarray) -> tuple[bool, list[float]]:
    d = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
    med = float(np.median(d))
    return bool(((d > 25) & (d > 4 * med)).any()), [round(float(x), 2) for x in d]


def cuts() -> None:
    """Apply the cut rule to Xiph-B (exclusion) and audit already-used sets (report only)."""
    ledger = json.loads((REPO / 'research/data/ledger.json').read_text())
    report = {}
    for name, folder in (('xiph-b', 'xiph-b'), ('xiph', 'xiph'), ('snu-film', 'snu-film')):
        src = next(s for s in ledger['sources'] if s['name'] == name)
        for c in src['sequences']:
            cut, d = _has_cut(np.load(EXTERNAL / folder / c['file'])['frames'])
            report.setdefault(name, {})[c['id']] = dict(cut=cut, max_diff=max(d), median_diff=float(np.median(d)))
    report['xiph-b_excluded'] = sorted(k for k, v in report['xiph-b'].items() if v['cut'])
    report['rule'] = 'cut = consecutive mean |dRGB| > 25 and > 4x clip median; Xiph-B clips with a cut are excluded (v3 amendment 1)'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'xiph-b-cuts.json').write_text(json.dumps(report, indent=2))
    for name in ('xiph-b', 'xiph', 'snu-film'):
        flagged = [k for k, v in report[name].items() if v['cut']]
        print(name, 'clips with cuts:', flagged)


if __name__ == '__main__':
    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else 'xiph-b'
    {'devpass': lambda: devpass(), 'fit': fit, 'testpass': lambda: testpass(arg), 'judge': lambda: judge(arg), 'cuts': cuts}[cmd]()
