"""Phase 2 experiments: recoverability prediction (C2, gate G2) and calibrated risk masks (C3, gates G3/G4).

Inputs: LacunaBench v1 rows (research/results/lacunabench-v1/rows.csv) and error
maps ($LACUNA_EXTERNAL/results/lacunabench-v1/<clip>.npz). Features use
observed frames only (research/lacunabench/recoverability.py). Models are fit
on the dev split (20 YouTube clips) and evaluated once on the test split (11
GoPro clips).

    .venv/bin/python research/scripts/recoverability_v1.py features   # observation-only features per clip
    .venv/bin/python research/scripts/recoverability_v1.py evaluate   # G2, G3, G4 + report
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
BENCH = EXTERNAL / 'results/lacunabench-v1'
FEATS = EXTERNAL / 'results/recoverability-v1'
OUT = REPO / 'research/results/recoverability-v1'
TASKS = ('gap1', 'gap2', 'gap4', 'gap8', 'gap16', 'gap32', 'bernoulli', 'gilbert_elliott', 'gop_chain')
METHODS = ('hold', 'linear', 'flow', 'flow_quadratic', 'rife')
TAU = 20.0      # a pixel is "bad" when mean |error| > 20 / 255 (80x45 pooled map)
ALPHA = 0.10    # target miss rate of bad pixels
PIXELS_PER_FRAME = 300
SEED = 20261120


def clip_features(args: tuple) -> str:
    clip, file = args[:2]
    data_dir, bench_dir, feats_dir = (args[2:] + (None, None, None))[:3] if len(args) > 2 else (None, None, None)
    data_dir, bench_dir, feats_dir = data_dir or EXTERNAL / 'snu-film', bench_dir or BENCH, feats_dir or FEATS
    from research.lacunabench.recoverability import BracketFeatures, brackets, frame_summary
    frames = np.load(Path(data_dir) / file)['frames']
    bench = np.load(Path(bench_dir) / f'{clip}.npz')
    out = {}
    for task in TASKS:
        missing = bench[f'{task}/missing']
        maps, order = [], []
        for left, right, l2, r2, idx in brackets(missing):
            bf = BracketFeatures(frames, left, right, l2, r2)
            for i in idx:
                maps.append(bf.maps(i))
                order.append(i)
        assert order == list(np.flatnonzero(missing)), (clip, task)  # aligns with error-map order
        maps = np.stack(maps)
        out[f'{task}/maps'] = maps.astype(np.float16)
        out[f'{task}/summary'] = np.stack([frame_summary(m) for m in maps]).astype(np.float32)
        out[f'{task}/frames'] = np.array(order)
    Path(feats_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(feats_dir) / f'{clip}.npz', **out)
    return clip


def features(workers: int = 4) -> None:
    jobs = [(c['id'], c['file']) for split in ('dev', 'test') for c in SPLITS[split]]
    with ProcessPoolExecutor(workers) as pool:
        for clip in pool.map(clip_features, jobs):
            print('features', clip, flush=True)


# --------------------------------------------------------------------------- data assembly

def load_psnr(rows_csv: Path | None = None) -> dict:
    table = {}
    with (rows_csv or REPO / 'research/results/lacunabench-v1/rows.csv').open() as f:
        for r in csv.DictReader(f):
            table[(r['clip'], r['task'], int(r['frame']), r['method'])] = min(float(r['psnr']), 60.0)
    return table


def assemble(split: str | list[dict], psnr: dict, bench_dir: Path | None = None, feats_dir: Path | None = None,
             methods: tuple[str, ...] = METHODS) -> dict:
    """Per-frame and per-pixel arrays for one split (name) or an explicit list of clip entries."""
    clips = SPLITS[split] if isinstance(split, str) else split
    bench_dir, feats_dir = bench_dir or BENCH, feats_dir or FEATS
    from research.lacunabench.recoverability import CHANNELS, PAIR_ONLY
    pair_idx = [CHANNELS.index(c) for c in PAIR_ONLY]
    frames = dict(clip=[], task=[], run=[], summary=[], pair_summary=[], maps=[], err={m: [] for m in methods}, psnr={m: [] for m in methods})
    from research.lacunabench.recoverability import frame_summary
    for c in clips:
        clip = c['id']
        fz, bz = np.load(Path(feats_dir) / f'{clip}.npz'), np.load(Path(bench_dir) / f'{clip}.npz')
        for task in TASKS:
            maps = fz[f'{task}/maps'].astype(np.float32)
            idx = fz[f'{task}/frames']
            missing = bz[f'{task}/missing']
            run_len = _run_lengths(missing)
            for k, i in enumerate(idx):
                frames['clip'].append(clip)
                frames['task'].append(task)
                frames['run'].append(int(run_len[i]))
                frames['summary'].append(fz[f'{task}/summary'][k])
                frames['pair_summary'].append(frame_summary(maps[k], PAIR_ONLY))
                frames['maps'].append(maps[k])
                for m in methods:
                    frames['err'][m].append(bz[f'{task}/{m}'][k].astype(np.float32))
                    frames['psnr'][m].append(psnr[(clip, task, int(i), m)])
    out = {k: np.array(v) if k not in ('err', 'psnr') else {m: np.array(x) for m, x in v.items()} for k, v in frames.items()}
    out['pair_idx'] = pair_idx
    return out


def _run_lengths(missing: np.ndarray) -> np.ndarray:
    n, out, i = len(missing), np.zeros(len(missing), int), 0
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def pixel_table(data: dict, method: str, channels: list[int] | None, rng=None, per_frame: int | None = PIXELS_PER_FRAME):
    """Rows = pixels: per-pixel channels plus the frame summary, target = abs error."""
    maps, summ = data['maps'], (data['summary'] if channels is None else data['pair_summary'])
    n, c, h, w = maps.shape
    sel = slice(None) if channels is None else channels
    X, y, frame_id = [], [], []
    for k in range(n):
        px = maps[k][sel].reshape(len(range(c)) if channels is None else len(channels), -1).T
        target = data['err'][method][k].reshape(-1)
        if per_frame is not None and rng is not None:
            pick = rng.choice(len(target), per_frame, replace=False)
            px, target = px[pick], target[pick]
        X.append(np.hstack([px, np.repeat(summ[k][None], len(px), 0)]))
        y.append(target)
        frame_id.append(np.full(len(target), k))
    return np.vstack(X), np.concatenate(y), np.concatenate(frame_id)


def model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=.06, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)


# --------------------------------------------------------------------------- evaluation

def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def per_clip_rho(clips, pred, actual) -> dict[str, float]:
    return {c: spearman(pred[clips == c], actual[clips == c]) for c in np.unique(clips)}


def evaluate() -> None:
    from sklearn.metrics import roc_auc_score
    from backend.stats import bootstrap_mean, paired
    from research.lacunabench.conformal import Unit, calibrate, evaluate as crc_eval
    rng = np.random.default_rng(SEED)
    psnr = load_psnr()
    dev, test = assemble('dev', psnr), assemble('test', psnr)
    report, lines = dict(G2={}, transfer={}, pixel={}, G3={}, G4={}), []

    # ---- C2 per-frame: predict PSNR of each method from observations only.
    lines += ['# Recoverability (C2) and calibrated risk masks (C3): v1 results', '',
              'Models are fit on dev (20 YouTube clips) and evaluated once on test (11 GoPro clips). Inputs come from observed frames only. The pair-only baseline sees the two anchor frames but not the gap length, the timestep or the outer anchors (a feature-based analogue of VFI-DPA\'s pair-difficulty idea, not its CNN). Clips are the unit for CIs.', '',
              '## C2: per-frame recoverability (Spearman ρ between predicted and actual PSNR on test)', '',
              '| Target method | Full model ρ (pooled) | Pair-only ρ (pooled) | Full ρ per clip, mean [95% CI] | Δρ vs pair-only per clip [95% CI] | Clips better |', '|---|---:|---:|---|---|---:|']
    frame_models = {}
    for m in METHODS:
        full, pair = model().fit(dev['summary'], dev['psnr'][m]), model().fit(dev['pair_summary'], dev['psnr'][m])
        frame_models[m] = full
        pf, pp = full.predict(test['summary']), pair.predict(test['pair_summary'])
        rho_full, rho_pair = spearman(pf, test['psnr'][m]), spearman(pp, test['psnr'][m])
        cf, cp = per_clip_rho(test['clip'], pf, test['psnr'][m]), per_clip_rho(test['clip'], pp, test['psnr'][m])
        ci = bootstrap_mean(list(cf.values()))
        d = paired(cf, cp, ('full', 'pair_only'))
        report['G2'][m] = dict(rho_pooled=rho_full, rho_pair_pooled=rho_pair, per_clip=ci.to_dict(), vs_pair=d.to_dict())
        lines.append(f'| {m} | {rho_full:.3f} | {rho_pair:.3f} | {ci.mean:.3f} [{ci.low:.3f}, {ci.high:.3f}] | {d.delta.mean:+.3f} [{d.delta.low:+.3f}, {d.delta.high:+.3f}] | {d.wins}/{d.delta.n_scenes} |')
    g2m = report['G2']['rife']
    g2_pass = g2m['rho_pooled'] >= .6 and g2m['vs_pair']['delta']['low'] > 0
    report['G2']['gate'] = dict(target='rife', passed=g2_pass, rule='pooled per-frame Spearman >= 0.6 on test and paired per-clip CI vs pair-only excludes 0')
    lines += ['', f"**Gate G2 (pre-registered, judged on RIFE, the strongest reconstructor): {'PASSED' if g2_pass else 'FAILED'}** (pooled ρ = {g2m['rho_pooled']:.3f}; per-clip Δρ vs pair-only CI [{g2m['vs_pair']['delta']['low']:+.3f}, {g2m['vs_pair']['delta']['high']:+.3f}]).", '']

    # ---- Transfer: a model trained on one method's errors ranks another method's errors.
    lines += ['## C2 transfer across reconstructors (pooled Spearman ρ on test)', '', '| Trained on → evaluated on | ' + ' | '.join(METHODS) + ' |', '|---|' + '---:|' * len(METHODS)]
    for src in ('flow', 'rife'):
        pred = frame_models[src].predict(test['summary'])
        vals = {dst: spearman(pred, test['psnr'][dst]) for dst in METHODS}
        report['transfer'][src] = vals
        lines.append(f'| {src} | ' + ' | '.join(f'{vals[d]:.3f}' for d in METHODS) + ' |')
    lines.append('')

    # ---- C2 per-pixel + C3 risk masks for flow and RIFE.
    lines += ['## C2 per-pixel and C3 conformal risk masks', '',
              f'A pixel is bad when its pooled mean absolute error exceeds {TAU:.0f}/255. Masks are calibrated with conformal risk control at α = {ALPHA} (target: miss at most 10% of bad pixels), with clips as the exchangeable units (20 dev clips; finite-sample term 1/21). Area is the fraction of missing-frame pixels flagged; smaller is better at equal coverage.', '',
              '| Method | Score | Pixel AUROC (bad vs good) | Test coverage of bad pixels | Flagged area | Area vs pair-only |', '|---|---|---:|---:|---:|---:|']
    for m in ('flow', 'rife'):
        Xf, yf, _ = pixel_table(dev, m, None, rng)
        Xp, yp, _ = pixel_table(dev, m, dev['pair_idx'], rng)
        full_px, pair_px = model().fit(Xf, yf), model().fit(Xp, yp)
        Tf, ty, tid = pixel_table(test, m, None, per_frame=None)
        Tp, _, _ = pixel_table(test, m, test['pair_idx'], per_frame=None)
        Df, dy, did = pixel_table(dev, m, None, per_frame=None)
        Dp, _, _ = pixel_table(dev, m, dev['pair_idx'], per_frame=None)
        scores = {'conditioned (full)': (full_px.predict(Df), full_px.predict(Tf)),
                  'pair-only': (pair_px.predict(Dp), pair_px.predict(Tp)),
                  'anchor difference (heuristic)': (Df[:, 2], Tf[:, 2])}
        pixels_per_frame = test['maps'].shape[2] * test['maps'].shape[3]
        dev_clip = np.repeat(dev['clip'], pixels_per_frame)
        test_clip = np.repeat(test['clip'], pixels_per_frame)
        dev_run = np.repeat(dev['run'], pixels_per_frame)
        test_run = np.repeat(test['run'], pixels_per_frame)
        test_task = np.repeat(test['task'], pixels_per_frame)
        report['pixel'][m], report['G3'][m] = {}, {}
        area = {}
        for name, (s_dev, s_test) in scores.items():
            auc = float(roc_auc_score(ty > TAU, s_test))
            cal = [Unit(s_dev[dev_clip == c], dy[dev_clip == c]) for c in np.unique(dev_clip)]
            lam = calibrate(cal, ALPHA, TAU)
            res = crc_eval([Unit(s_test[test_clip == c], ty[test_clip == c]) for c in np.unique(test_clip)], lam, TAU)
            area[name] = res['flagged_fraction']
            report['pixel'][m][name] = dict(auroc=auc, lambda_hat=lam, coverage=res['coverage'], flagged_fraction=res['flagged_fraction'], per_clip_risk=res['per_unit_risk'])
        for name in scores:
            r = report['pixel'][m][name]
            rel = '—' if name == 'pair-only' else f"{(r['flagged_fraction'] / area['pair-only'] - 1) * 100:+.1f}%"
            lines.append(f"| {m} | {name} | {r['auroc']:.3f} | {r['coverage']:.3f} | {r['flagged_fraction']:.3f} | {rel} |")
        # Mondrian by gap stratum (short <= 4 frames, long > 4) for the conditioned score; check every gap/process stratum.
        s_dev, s_test = scores['conditioned (full)']
        strata = {}
        for stratum, sel_dev, sel_test in [('short (≤4)', dev_run <= 4, test_run <= 4), ('long (>4)', dev_run > 4, test_run > 4)]:
            cal = [Unit(s_dev[(dev_clip == c) & sel_dev], dy[(dev_clip == c) & sel_dev]) for c in np.unique(dev_clip)]
            lam = calibrate(cal, ALPHA, TAU)
            for task in TASKS:
                sel = sel_test & (test_task == task)
                if not sel.any():
                    continue
                res = crc_eval([Unit(s_test[(test_clip == c) & sel], ty[(test_clip == c) & sel]) for c in np.unique(test_clip[sel])], lam, TAU)
                strata[f'{task} / {stratum}'] = dict(coverage=res['coverage'], flagged_fraction=res['flagged_fraction'], units=res['units_with_bad_pixels'])
        report['G3'][m] = strata
        reduction = 1 - area['conditioned (full)'] / area['pair-only']
        report['G4'][m] = dict(area_conditioned=area['conditioned (full)'], area_pair_only=area['pair-only'], reduction=reduction,
                               coverage_conditioned=report['pixel'][m]['conditioned (full)']['coverage'],
                               coverage_pair_only=report['pixel'][m]['pair-only']['coverage'])
    lines.append('')
    g4 = report['G4']['rife']
    g4_pass = g4['reduction'] >= .15 and min(g4['coverage_conditioned'], g4['coverage_pair_only']) >= 1 - ALPHA - .02
    report['G4']['gate'] = dict(target='rife', passed=g4_pass, rule='>= 15% smaller flagged area than pair-only CRC at equal target coverage (both >= 1-alpha-0.02 on test)')
    lines += [f"**Gate G4 (RIFE): {'PASSED' if g4_pass else 'FAILED'}**: the conditioned mask flags {g4['area_conditioned']:.3f} of pixels vs {g4['area_pair_only']:.3f} for pair-only ({g4['reduction']*100:.1f}% smaller), with test coverage {g4['coverage_conditioned']:.3f} vs {g4['coverage_pair_only']:.3f}.", '']

    lines += ['## C3 coverage by stratum (conditioned score, Mondrian by gap length, RIFE)', '', '| Stratum | Coverage | Flagged area | Clips with bad pixels |', '|---|---:|---:|---:|']
    worst = 1.0
    for k, v in report['G3']['rife'].items():
        lines.append(f"| {k} | {v['coverage']:.3f} | {v['flagged_fraction']:.3f} | {v['units']} |")
        if v['units'] >= 3:
            worst = min(worst, v['coverage'])
    g3_pass = worst >= 1 - ALPHA - .02
    report['G3']['gate'] = dict(target='rife', passed=g3_pass, worst_coverage=worst, rule='coverage >= 1-alpha-0.02 in every stratum with >= 3 test clips containing bad pixels')
    lines += ['', f"**Gate G3 (RIFE): {'PASSED' if g3_pass else 'FAILED'}**: worst stratum coverage {worst:.3f} (target ≥ {1 - ALPHA - .02:.2f}). Strata with fewer than 3 test clips containing bad pixels are shown but not gated.", '',
              'Caveats: 11 test clips give wide uncertainty on coverage. The guarantee is marginal over clips, not per frame. τ and α were fixed in this script before the test split was evaluated.']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2, default=float))
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'features': features, 'evaluate': evaluate}[sys.argv[1]]()
