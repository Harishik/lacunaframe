"""Conformal risk control for per-pixel risk masks (contribution C3).

A risk mask flags pixels whose score is >= lambda. A pixel is *bad* when its
reconstruction error exceeds tau. The loss of one calibration unit is its
false-negative rate: bad pixels left unflagged / bad pixels, pooled over the
unit's frames (0 if it has no bad pixels). Lowering lambda flags more pixels,
so the loss is non-increasing as lambda decreases. Conformal risk control
(Angelopoulos et al., arXiv 2208.02814) picks

    lambda_hat = max { lambda : n/(n+1) * mean_i L_i(lambda) + B/(n+1) <= alpha },  B = 1,

which guarantees E[L_test(lambda_hat)] <= alpha when calibration and test units
are exchangeable. **Units are clips, not pixels or frames**, so the guarantee
does not rely on independence between neighbouring pixels or frames.

Mondrian (stratified) calibration fits a separate lambda per stratum (e.g. gap
length), giving the guarantee within each stratum.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Unit:
    """One exchangeable calibration/test unit (a clip): flattened scores and errors."""
    scores: np.ndarray
    errors: np.ndarray


def unit_loss(unit: Unit, lam: float, tau: float) -> float:
    bad = unit.errors > tau
    if not bad.any():
        return 0.0
    return float(np.mean(unit.scores[bad] < lam))


def _grid(units: list[Unit], size: int = 400) -> np.ndarray:
    pooled = np.concatenate([u.scores for u in units])
    qs = np.unique(np.quantile(pooled, np.linspace(0, 1, size)))
    return np.concatenate([qs, [np.inf]])[::-1]  # descending: smallest mask first


def calibrate(units: list[Unit], alpha: float, tau: float, bound: float = 1.0) -> float:
    """Return lambda_hat. If no finite lambda meets the target, return -inf (flag everything)."""
    if not units:
        raise ValueError('Need at least one calibration unit')
    n = len(units)
    # Scan from the smallest mask (largest lambda) downwards; risk only falls as lambda falls,
    # so the first feasible lambda is the largest one.
    for lam in _grid(units):
        risk = np.mean([unit_loss(u, lam, tau) for u in units])
        if n / (n + 1) * risk + bound / (n + 1) <= alpha:
            return float(lam)
    return -np.inf


def evaluate(units: list[Unit], lam: float, tau: float) -> dict:
    """Realised mean FNR (risk), coverage of bad pixels and flagged-area fraction on test units."""
    losses = [unit_loss(u, lam, tau) for u in units]
    flagged = [float(np.mean(u.scores >= lam)) for u in units]
    has_bad = [bool((u.errors > tau).any()) for u in units]
    return dict(risk=float(np.mean(losses)), coverage=1 - float(np.mean(losses)), flagged_fraction=float(np.mean(flagged)),
                units=len(units), units_with_bad_pixels=int(sum(has_bad)), per_unit_risk=losses)


def calibrate_mondrian(units: dict[str, list[Unit]], alpha: float, tau: float) -> dict[str, float]:
    """Separate lambda per stratum; each stratum's units must be exchangeable across splits."""
    return {k: calibrate(v, alpha, tau) for k, v in units.items() if v}
