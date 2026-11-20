"""Recoverability-guided fusion (RGF), protocol v3 method A.

Several reconstructors produce candidate frames. Per-candidate error predictors,
fit on dev only, estimate each candidate's pooled per-pixel error from
observation-only recoverability features plus *candidate-disagreement*
features (computed from the candidates themselves, never the reference). The
fused frame is a per-pixel softmax blend: w_k = softmax(-e_k / T), computed on the
80x45 grid and bilinearly upsampled to full resolution.
"""
from __future__ import annotations
import cv2
import numpy as np
from .recoverability import MAP

CANDIDATES = ('rife', 'amt_s_gopro', 'ifrnet_gopro', 'flow', 'linear')
DIS_CHANNELS = tuple(f'dev_{c}' for c in CANDIDATES) + ('cand_std',)


def _pool(x: np.ndarray) -> np.ndarray:
    return cv2.resize(x.astype(np.float32), MAP, interpolation=cv2.INTER_AREA)


def disagreement(stack: np.ndarray) -> np.ndarray:
    """(K, H, W, 3) uint8 candidates for one frame -> (K + 1, 45, 80) disagreement maps.

    Channel k: mean |candidate_k - per-pixel candidate median| over RGB; last channel:
    per-pixel standard deviation across candidates (mean over RGB).
    """
    x = stack.astype(np.float32)
    median = np.median(x, axis=0)
    dev = [_pool(np.abs(c - median).mean(-1)) for c in x]
    std = _pool(x.std(axis=0).mean(-1))
    return np.stack(dev + [std])


def pooled_error(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Same target as the v1 error maps: mean |error| over RGB, area-pooled to 80x45."""
    return _pool(np.abs(ref.astype(np.float32) - est.astype(np.float32)).mean(-1))


def pixel_features(rec_maps: np.ndarray, dis_maps: np.ndarray) -> np.ndarray:
    """Per-pixel rows for one frame: 9 recoverability + K+1 disagreement channels, plus frame summaries."""
    maps = np.concatenate([rec_maps, dis_maps]).astype(np.float32)          # (C, 45, 80)
    flat = maps.reshape(len(maps), -1)
    summary = np.concatenate([flat.mean(1), np.percentile(flat, 90, axis=1)])
    return np.hstack([flat.T, np.repeat(summary[None], flat.shape[1], 0)])


def fuse(stack: np.ndarray, predicted: np.ndarray, temperature: float) -> np.ndarray:
    """Blend (K, H, W, 3) candidates with softmax(-predicted / T) weights from (K, 45, 80) predicted errors."""
    k, h, w = stack.shape[0], stack.shape[1], stack.shape[2]
    logits = -predicted.astype(np.float64) / temperature
    logits -= logits.max(axis=0, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=0, keepdims=True)
    full = np.stack([cv2.resize(wk.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) for wk in weights])
    full /= full.sum(axis=0, keepdims=True)
    out = (full[..., None] * stack.astype(np.float32)).sum(axis=0)
    return np.clip(out, 0, 255).round().astype(np.uint8)
