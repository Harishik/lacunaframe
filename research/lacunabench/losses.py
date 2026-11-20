"""Seeded temporal loss processes for protocol v1 task T3.

Every process returns a boolean `missing` vector. `protect` keeps the first and
last `protect` samples observed so that every gap is bracketed by at least that
many anchors (the protocol scores unbracketed boundaries separately).
"""
from __future__ import annotations
import numpy as np


def _protect(missing: np.ndarray, protect: int) -> np.ndarray:
    if protect:
        missing[:protect] = False
        missing[-protect:] = False
    return missing


def contiguous(n: int, start: int, gap: int, protect: int = 2) -> np.ndarray:
    """One run of `gap` missing samples starting at `start`."""
    if start < protect or start + gap > n - protect:
        raise ValueError('Gap must leave `protect` anchors on both sides')
    missing = np.zeros(n, bool)
    missing[start:start + gap] = True
    return missing


def bernoulli(n: int, rate: float, rng: np.random.Generator, protect: int = 2) -> np.ndarray:
    """Independent drops with probability `rate`."""
    return _protect(rng.random(n) < rate, protect)


def gilbert_elliott(n: int, rate: float, burst: float, rng: np.random.Generator, protect: int = 2) -> np.ndarray:
    """Two-state Markov burst loss: every sample in the Bad state is lost, none in Good.

    Mean burst length is `burst` (P(B->G) = 1/burst); the stationary Bad probability
    equals `rate`, so the expected loss rate matches the Bernoulli process.
    """
    if not 0 < rate < 1 or burst < 1:
        raise ValueError('0 < rate < 1 and burst >= 1 required')
    p_bg = 1 / burst
    p_gb = rate * p_bg / (1 - rate)
    if p_gb > 1:
        raise ValueError('rate/burst combination is infeasible')
    missing = np.zeros(n, bool)
    bad = rng.random() < rate
    for i in range(n):
        missing[i] = bad
        bad = (rng.random() >= p_bg) if bad else (rng.random() < p_gb)
    return _protect(missing, protect)


def gop_chain(n: int, frame_loss: float, gop: int, rng: np.random.Generator, protect: int = 2) -> np.ndarray:
    """Codec-style dependency loss: GOPs of `gop` frames (I P P ...). A lost frame makes
    every later frame in its GOP undecodable until the next I-frame."""
    if gop < 1:
        raise ValueError('gop >= 1 required')
    missing = np.zeros(n, bool)
    for start in range(0, n, gop):
        for i in range(start, min(n, start + gop)):
            if rng.random() < frame_loss:
                missing[i:min(n, start + gop)] = True
                break
    return _protect(missing, protect)


def gop_frame_loss_for_rate(rate: float, gop: int) -> float:
    """Per-frame loss q giving expected loss fraction `rate` for gop_chain (ignoring protection).

    Expected lost fraction f(q) = 1 - (1 - (1-q)^(G+1)) / (G q) for a GOP of G frames
    (frame k, 0-based, is lost iff any of frames 0..k is lost). Solved by bisection.
    """
    def lost(q: float) -> float:
        return float(np.mean([1 - (1 - q) ** (k + 1) for k in range(gop)]))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if lost(mid) < rate else (lo, mid)
    return (lo + hi) / 2


def runs(missing: np.ndarray) -> list[int]:
    """Lengths of contiguous missing runs."""
    lengths, current = [], 0
    for m in missing:
        if m:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths
