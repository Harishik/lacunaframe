"""Conformal risk control: guarantee on exchangeable synthetic data, monotonicity, edges."""
import numpy as np
from research.lacunabench.conformal import Unit, calibrate, evaluate, unit_loss


def synthetic_units(rng, n, informative=True):
    units = []
    for _ in range(n):
        difficulty = rng.gamma(2, 1)  # clip-level random effect: units are exchangeable, pixels are not independent
        score = rng.gamma(2, 1, 600) * difficulty
        noise = rng.normal(0, 1.5 if informative else 50, 600)
        units.append(Unit(score, 8 * score + noise if informative else rng.gamma(2, 8, 600)))
    return units


def test_loss_is_monotone_in_lambda():
    u = synthetic_units(np.random.default_rng(0), 1)[0]
    lams = np.linspace(0, 20, 50)
    losses = [unit_loss(u, l, 20) for l in lams]
    assert all(a <= b + 1e-12 for a, b in zip(losses, losses[1:]))


def test_marginal_risk_guarantee_holds_over_random_splits():
    rng = np.random.default_rng(1)
    alpha, tau, realised = .2, 20, []
    for _ in range(150):
        units = synthetic_units(rng, 41)
        lam = calibrate(units[:40], alpha, tau)
        realised.append(unit_loss(units[40], lam, tau))
    # E[L_test] <= alpha; 150 draws give a standard error of about 0.03.
    assert np.mean(realised) <= alpha + .06


def test_informative_scores_give_smaller_masks_than_uninformative():
    rng = np.random.default_rng(2)
    good, bad = synthetic_units(rng, 30), synthetic_units(rng, 30, informative=False)
    for units in (good, bad):
        for u in units:
            u.errors = np.asarray(u.errors)
    lam_good = calibrate(good[:20], .2, 20)
    area_good = evaluate(good[20:], lam_good, 20)['flagged_fraction']
    # Uninformative: random scores against the same errors.
    shuffled = [Unit(rng.permutation(u.scores), u.errors) for u in good]
    lam_rand = calibrate(shuffled[:20], .2, 20)
    area_rand = evaluate(shuffled[20:], lam_rand, 20)['flagged_fraction']
    assert area_good < area_rand


def test_impossible_target_flags_everything():
    units = synthetic_units(np.random.default_rng(3), 3)
    assert calibrate(units, alpha=.1, tau=20) == -np.inf  # B/(n+1) = 0.25 > alpha
