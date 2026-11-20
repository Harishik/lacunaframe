"""Protocol v2 confirmation on the untouched Xiph set: G2', G3', G4' plus descriptive baselines.

Everything that is fit (recoverability models, conformal lambdas, tau') comes
from the v1 dev split only. Xiph is scored once.

    .venv/bin/python research/scripts/confirm_v2.py features
    .venv/bin/python research/scripts/confirm_v2.py evaluate
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
from research.scripts import recoverability_v1 as v1

EXTERNAL = v1.EXTERNAL
XBENCH = EXTERNAL / 'results/lacunabench-v1-xiph'
XFEATS = EXTERNAL / 'results/recoverability-v2-xiph'
XROWS = REPO / 'research/results/lacunabench-v1-xiph/rows.csv'
OUT = REPO / 'research/results/confirmation-v2'
TAU_PRIME = json.loads((REPO / 'research/protocols/v2-tau-prime.json').read_text())['tau_prime']
ALPHA = 0.10


def xiph_clips() -> list[dict]:
    ledger = json.loads((REPO / 'research/data/ledger.json').read_text())
    return next(s for s in ledger['sources'] if s['name'] == 'xiph')['sequences']


def features(workers: int = 4) -> None:
    jobs = [(c['id'], c['file'], EXTERNAL / 'xiph', XBENCH, XFEATS) for c in xiph_clips()]
    with ProcessPoolExecutor(workers) as pool:
        for clip in pool.map(v1.clip_features, jobs):
            print('features', clip, flush=True)


def _units(scores, errors, clip_ids, select=None):
    from research.lacunabench.conformal import Unit
    keep = np.ones(len(scores), bool) if select is None else select
    return [Unit(scores[(clip_ids == c) & keep], errors[(clip_ids == c) & keep]) for c in np.unique(clip_ids[keep])]


def evaluate() -> None:
    from backend.stats import bootstrap_mean, holm, paired, scene_means
    from research.lacunabench.conformal import calibrate, evaluate as crc_eval
    rng = np.random.default_rng(v1.SEED)
    dev = v1.assemble('dev', v1.load_psnr(), methods=('rife',))
    xiph = v1.assemble(xiph_clips(), v1.load_psnr(XROWS), XBENCH, XFEATS, methods=('rife',))
    report, lines = {}, ['# Protocol v2 confirmation on untouched Xiph 1080p (12 clips)', '',
                         'All models, conformal thresholds and τ′ were fit on the v1 dev split (20 YouTube clips) and frozen before Xiph was scored. Xiph differs in camera, content and frame rate (25 fps). Clips are the unit for CIs.', '']

    # ---- G2': per-frame recoverability transfer.
    full = v1.model().fit(dev['summary'], dev['psnr']['rife'])
    pair = v1.model().fit(dev['pair_summary'], dev['psnr']['rife'])
    pf, pp = full.predict(xiph['summary']), pair.predict(xiph['pair_summary'])
    rho, rho_pair = v1.spearman(pf, xiph['psnr']['rife']), v1.spearman(pp, xiph['psnr']['rife'])
    cf, cp = v1.per_clip_rho(xiph['clip'], pf, xiph['psnr']['rife']), v1.per_clip_rho(xiph['clip'], pp, xiph['psnr']['rife'])
    d = paired(cf, cp, ('full', 'pair_only'))
    g2 = rho >= .6 and d.delta.low > 0
    report['G2_prime'] = dict(rho=rho, rho_pair=rho_pair, per_clip=bootstrap_mean(list(cf.values())).to_dict(), vs_pair=d.to_dict(), passed=g2)
    lines += ['## G2′ recoverability transfer (RIFE)', '',
              f'Pooled per-frame Spearman ρ = **{rho:.3f}** (pair-only {rho_pair:.3f}). Per-clip Δρ vs pair-only = {d.delta.mean:+.3f} [{d.delta.low:+.3f}, {d.delta.high:+.3f}], {d.wins}/{d.delta.n_scenes} clips better.',
              f"**G2′ {'PASSED' if g2 else 'FAILED'}.**", '']

    # ---- Pixel models (dev) and scores on both sets.
    Xf, yf, _ = v1.pixel_table(dev, 'rife', None, rng)
    Xp, yp, _ = v1.pixel_table(dev, 'rife', dev['pair_idx'], rng)
    full_px, pair_px = v1.model().fit(Xf, yf), v1.model().fit(Xp, yp)
    Df, dy, _ = v1.pixel_table(dev, 'rife', None, per_frame=None)
    Dp, _, _ = v1.pixel_table(dev, 'rife', dev['pair_idx'], per_frame=None)
    Tf, ty, _ = v1.pixel_table(xiph, 'rife', None, per_frame=None)
    Tp, _, _ = v1.pixel_table(xiph, 'rife', xiph['pair_idx'], per_frame=None)
    ppf = dev['maps'].shape[2] * dev['maps'].shape[3]
    dclip, xclip = np.repeat(dev['clip'], ppf), np.repeat(xiph['clip'], ppf)
    drun, xrun = np.repeat(dev['run'], ppf), np.repeat(xiph['run'], ppf)
    xtask = np.repeat(xiph['task'], ppf)
    s_dev_full, s_x_full = full_px.predict(Df), full_px.predict(Tf)
    s_dev_pair, s_x_pair = pair_px.predict(Dp), pair_px.predict(Tp)

    # ---- G3': v1 dev Mondrian lambda (tau = 20) applied to Xiph strata.
    strata, worst = {}, 1.0
    for name, sel_d, sel_x in [('short (≤4)', drun <= 4, xrun <= 4), ('long (>4)', drun > 4, xrun > 4)]:
        lam = calibrate(_units(s_dev_full, dy, dclip, sel_d), ALPHA, v1.TAU)
        for task in v1.TASKS:
            sel = sel_x & (xtask == task)
            if not sel.any():
                continue
            r = crc_eval(_units(s_x_full, ty, xclip, sel), lam, v1.TAU)
            strata[f'{task} / {name}'] = dict(coverage=r['coverage'], flagged=r['flagged_fraction'], units=r['units_with_bad_pixels'])
            if r['units_with_bad_pixels'] >= 3:
                worst = min(worst, r['coverage'])
    g3 = worst >= 1 - ALPHA - .02
    report['G3_prime'] = dict(strata=strata, worst=worst, passed=g3)
    lines += ['## G3′ calibration transfer (RIFE, τ = 20, dev-calibrated Mondrian λ)', '', '| Stratum | Coverage | Flagged area | Clips with bad pixels |', '|---|---:|---:|---:|']
    lines += [f"| {k} | {v['coverage']:.3f} | {v['flagged']:.3f} | {v['units']} |" for k, v in strata.items()]
    lines += ['', f"**G3′ {'PASSED' if g3 else 'FAILED'}**: worst gated stratum coverage {worst:.3f} (floor 0.88).", '']

    # ---- G4': revised tau' (dev 80th percentile), conditioned vs pair-only.
    res = {}
    for name, sd, sx in [('conditioned', s_dev_full, s_x_full), ('pair-only', s_dev_pair, s_x_pair)]:
        lam = calibrate(_units(sd, dy, dclip), ALPHA, TAU_PRIME)
        r = crc_eval(_units(sx, ty, xclip), lam, TAU_PRIME)
        res[name] = dict(lambda_hat=lam, coverage=r['coverage'], flagged=r['flagged_fraction'])
    ratio = res['conditioned']['flagged'] / res['pair-only']['flagged']
    g4 = ratio <= .85 and min(res['conditioned']['coverage'], res['pair-only']['coverage']) >= 1 - ALPHA - .02
    base_rate = float((ty > TAU_PRIME).mean())
    report['G4_prime'] = dict(tau_prime=TAU_PRIME, xiph_bad_rate=base_rate, **res, area_ratio=ratio, passed=g4)
    lines += ['## G4′ conditioning with dev-fixed τ′', '',
              f'τ′ = {TAU_PRIME:.2f} (dev 80th percentile; 20% of dev pixels bad). On Xiph, {base_rate*100:.1f}% of pixels are bad.', '',
              '| Score | Coverage | Flagged area |', '|---|---:|---:|',
              f"| conditioned | {res['conditioned']['coverage']:.3f} | {res['conditioned']['flagged']:.3f} |",
              f"| pair-only | {res['pair-only']['coverage']:.3f} | {res['pair-only']['flagged']:.3f} |", '',
              f"Area ratio conditioned / pair-only = {ratio:.3f} (pass ≤ 0.85). **G4′ {'PASSED' if g4 else 'FAILED'}.**", '']

    # ---- Descriptive baselines on Xiph (scene-clustered, vs RIFE 4.9).
    rows = []
    for path in (XROWS, REPO / 'research/results/lacunabench-v1-xiph-gopro/rows.csv'):
        if path.exists():
            with path.open() as f:
                rows += list(csv.DictReader(f))
    order = ('hold', 'linear', 'flow', 'rife', 'amt_s', 'ifrnet_s', 'ifrnet', 'amt_s_gopro', 'ifrnet_s_gopro', 'ifrnet_gopro')
    methods = [m for m in order if any(r['method'] == m for r in rows)]
    per = lambda m, sub, key: scene_means([dict(sequence=r['clip'], v=min(float(r[key]), 60.) if key == 'psnr' else float(r[key]))
                                           for r in sub if r['method'] == m and r[key] not in ('', 'None')], 'v')
    gaps = [r for r in rows if r['process'] == 'contiguous']
    lines += ['## Descriptive: all methods on Xiph (contiguous gaps 1–32, scene means)', '', 'Plain AMT/IFRNet names are Vimeo90K-trained (midpoint) checkpoints; `*_gopro` are the official arbitrary-time checkpoints (no scene or camera overlap with Xiph).', '', '| Method | PSNR (dB) | LPIPS | ΔPSNR vs RIFE 4.9 [95% CI] | Clips better | Holm p |', '|---|---:|---:|---|---:|---:|']
    rife = per('rife', gaps, 'psnr')
    comps = holm([paired(per(m, gaps, 'psnr'), rife, (m, 'rife')) for m in methods if m != 'rife'])
    report['baselines_xiph'] = {}
    for m in methods:
        p, l = per(m, gaps, 'psnr'), per(m, gaps, 'lpips')
        c = next((c for c in comps if c.method == m), None)
        report['baselines_xiph'][m] = dict(psnr=float(np.mean(list(p.values()))), lpips=float(np.mean(list(l.values()))), vs_rife=None if c is None else c.to_dict())
        extra = '— | — | —' if c is None else f'{c.delta.mean:+.2f} [{c.delta.low:+.2f}, {c.delta.high:+.2f}] | {c.wins}/{c.delta.n_scenes} | {"—" if c.holm_p is None else f"{c.holm_p:.4f}"}'
        lines.append(f"| {m} | {np.mean(list(p.values())):.2f} | {np.mean(list(l.values())):.3f} | {extra} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2, default=float))
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    {'features': features, 'evaluate': evaluate}[sys.argv[1]]()
