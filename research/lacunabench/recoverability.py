"""Observation-only recoverability features (contribution C2).

For a missing frame inside a bracket of observed anchors (left, right), and
optionally outer anchors (left2 < left, right2 > right), compute per-pixel
feature maps and per-frame summaries **without reconstructing anything and
without the reference**. Everything is computed at a working resolution of
WORK = 320x180 and pooled to the MAP grid (80x45) used for error targets.

Per-pixel channels
  flow_mag     |F_lr| / gap, pixels per frame (motion speed)
  inconsist    forward/backward flow disagreement |F_lr + F_rl(x + F_lr)|
  anchor_diff  mean |a - b| (photometric disagreement of the anchors)
  warp_resid   |a - warp(b, F_lr)| (what flow cannot explain: occlusion, lighting)
  texture      gradient magnitude of the left anchor
  divergence   |div F_lr| (expansion and contraction, a disocclusion proxy)
  accel        |acceleration| from the outer anchors (0 if unavailable)
  dist_anchor  frames to the nearest observed anchor (broadcast)
  log_gap      log2 of the anchor separation (broadcast)

Pair-only channels (the VFI-DPA-style baseline sees only these): anchor_diff,
flow_mag, inconsist, warp_resid, texture, divergence. No gap length, no
timestep and no outer anchors.
"""
from __future__ import annotations
import cv2
import numpy as np

WORK = (320, 180)
MAP = (80, 45)
CHANNELS = ('flow_mag', 'inconsist', 'anchor_diff', 'warp_resid', 'texture', 'divergence', 'accel', 'dist_anchor', 'log_gap')
PAIR_ONLY = ('flow_mag', 'inconsist', 'anchor_diff', 'warp_resid', 'texture', 'divergence')
_FARNEBACK = (.5, 3, 15, 3, 5, 1.2, 0)


def _gray(a: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(a, WORK, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(a, b, None, *_FARNEBACK)


def _grid() -> np.ndarray:
    w, h = WORK
    return np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)


def _pool(x: np.ndarray) -> np.ndarray:
    return cv2.resize(x.astype(np.float32), MAP, interpolation=cv2.INTER_AREA)


class BracketFeatures:
    """Pair features for one bracket, computed once and shared by every missing frame in it."""

    def __init__(self, frames: np.ndarray, left: int, right: int, left2: int | None, right2: int | None):
        self.left, self.right, self.gap = left, right, right - left
        ga, gb = _gray(frames[left]), _gray(frames[right])
        f_ab, f_ba = _flow(ga, gb), _flow(gb, ga)
        grid = _grid()
        end = grid + f_ab
        back = cv2.remap(f_ba, end, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_b = cv2.remap(gb.astype(np.float32), end, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dudx = cv2.Sobel(f_ab[..., 0], cv2.CV_32F, 1, 0, ksize=3) / 8
        dvdy = cv2.Sobel(f_ab[..., 1], cv2.CV_32F, 0, 1, ksize=3) / 8
        gx = cv2.Sobel(ga.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(ga.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8
        a_rgb = cv2.resize(frames[left], WORK, interpolation=cv2.INTER_AREA).astype(np.float32)
        b_rgb = cv2.resize(frames[right], WORK, interpolation=cv2.INTER_AREA).astype(np.float32)
        self.pair = dict(
            flow_mag=_pool(np.linalg.norm(f_ab, axis=-1) / self.gap),
            inconsist=_pool(np.linalg.norm(f_ab + back, axis=-1)),
            anchor_diff=_pool(np.abs(a_rgb - b_rgb).mean(-1)),
            warp_resid=_pool(np.abs(ga.astype(np.float32) - warped_b)),
            texture=_pool(np.hypot(gx, gy)),
            divergence=_pool(np.abs(dudx + dvdy)))
        # Constant-acceleration magnitude from outer anchors (pixels / frame^2), averaged over sides.
        accels = []
        if left2 is not None:
            s_prev = left - left2
            f_prev = _flow(ga, _gray(frames[left2]))
            accels.append(np.linalg.norm(2 * (f_ab * s_prev + f_prev * self.gap) / (s_prev * self.gap * (s_prev + self.gap)), axis=-1))
        if right2 is not None:
            s_next = right2 - right
            f_next = _flow(gb, _gray(frames[right2]))
            accels.append(np.linalg.norm(2 * (f_next * self.gap + f_ba * s_next) / (s_next * self.gap * (s_next + self.gap)), axis=-1))
        self.accel = _pool(np.mean(accels, axis=0)) if accels else np.zeros(MAP[::-1], np.float32)

    def maps(self, index: int) -> np.ndarray:
        """Per-pixel feature stack (C, 45, 80) for missing frame `index` in this bracket."""
        dist = min(index - self.left, self.right - index)
        shape = MAP[::-1]
        layers = {**self.pair, 'accel': self.accel,
                  'dist_anchor': np.full(shape, dist, np.float32), 'log_gap': np.full(shape, np.log2(self.gap), np.float32)}
        return np.stack([layers[c] for c in CHANNELS])


def frame_summary(maps: np.ndarray, channels: tuple[str, ...] = CHANNELS) -> np.ndarray:
    """Per-frame vector: mean and 90th percentile of each selected channel."""
    idx = [CHANNELS.index(c) for c in channels]
    sel = maps[idx].reshape(len(idx), -1)
    return np.concatenate([sel.mean(1), np.percentile(sel, 90, axis=1)])


def summary_names(channels: tuple[str, ...] = CHANNELS) -> list[str]:
    return [f'{c}_mean' for c in channels] + [f'{c}_p90' for c in channels]


def brackets(missing: np.ndarray) -> list[tuple[int, int, int | None, int | None, list[int]]]:
    """(left, right, left2, right2, missing indices) for every bracketed missing run."""
    observed = np.flatnonzero(~missing)
    out = []
    i, n = 0, len(missing)
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            before, after = observed[observed < i], observed[observed >= j]
            if len(before) and len(after):
                out.append((int(before[-1]), int(after[0]),
                            int(before[-2]) if len(before) > 1 else None,
                            int(after[1]) if len(after) > 1 else None, list(range(i, j))))
            i = j
        else:
            i += 1
    return out
