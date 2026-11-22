"""Training samples for LacunaNet (protocol v4 §4). Runs in the training environment.

GOPRO: multi-anchor samples with missing runs g in {1..16}, random target position,
outer anchors present with probability 0.8 (each side independently).
Vimeo90K triplets: g = 1 midpoint samples, outer anchors absent. Shards are streamed
one at a time (PNG bytes kept in RAM, parquet file deleted) and cycled.
"""
from __future__ import annotations
import io
import os
import random
import threading
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import torch
try:
    from .model import make_cond
    from .classical_flow import lk_flow
except ImportError:   # flat layout (Colab)
    from model import make_cond
    from classical_flow import lk_flow

CROP = 256
GOPRO_P, OUTER_P = 0.7, 0.8
VIMEO_REPO = 'https://huggingface.co/datasets/bijinc/vimeo-90k-medium/resolve/main/data/train-{:05d}-of-00025.parquet'


def _crop(frames: list[np.ndarray], rng: random.Random) -> list[np.ndarray]:
    h, w = frames[0].shape[:2]
    y, x = rng.randint(0, h - CROP), rng.randint(0, w - CROP)
    out = [f[y:y + CROP, x:x + CROP] for f in frames]
    if rng.random() < .5:
        out = [f[:, ::-1] for f in out]
    if rng.random() < .5:
        out = [f[::-1] for f in out]
    return out


class GoproSource:
    @classmethod
    def from_scenes(cls, scenes):
        obj = cls.__new__(cls); obj.scenes = list(scenes)
        return obj

    def __init__(self, folder: Path):
        self.scenes = [np.load(p)['frames'] for p in sorted(folder.glob('*.npz'))]
        if not self.scenes:
            raise FileNotFoundError(f'No GOPRO training scenes in {folder}')

    def sample(self, rng: random.Random, min_scale: float = 1.0):
        """Random sample (g ~ U{1..16}). min_scale < 1 resizes by s ~ U[min_scale, 1] before cropping."""
        si = rng.randrange(len(self.scenes)); n = len(self.scenes[si])
        g = rng.randint(1, min(16, n - 3))
        left = rng.randint(0, n - g - 2)
        sc = rng.uniform(max(min_scale, (CROP + 1) / min(self.scenes[si][0].shape[:2])), 1.0) if min_scale < 1.0 else 1.0
        return self.build(rng, si, left, g, sc)

    def build(self, rng: random.Random, si: int, left: int, g: int, sc: float = 1.0):
        """Sample with inner anchors (left, left + g + 1) of scene si, resized by sc (displacements scale by sc)."""
        seq = self.scenes[si]; n = len(seq)
        right = left + g + 1
        target = rng.randint(left + 1, right - 1)
        s1, s2 = rng.choice((1, 2)), rng.choice((1, 2))
        has_m2 = left - s1 >= 0 and rng.random() < OUTER_P
        has_p2 = right + s2 < n and rng.random() < OUTER_P
        frames = [seq[left], seq[right], seq[left - s1] if has_m2 else seq[left], seq[right + s2] if has_p2 else seq[right], seq[target]]
        if rng.random() < .5:   # temporal reversal: swap roles of the two sides
            frames = [frames[1], frames[0], frames[3], frames[2], frames[4]]
            has_m2, has_p2, s1, s2 = has_p2, has_m2, s2, s1
            target = left + right - target
        span = right - left
        t = (target - left) / span
        cond = make_cond(t, g, -t - s1 / span, 1 - t + s2 / span, has_m2, has_p2)
        if sc < 1.0:
            size = (int(round(frames[0].shape[1] * sc)), int(round(frames[0].shape[0] * sc)))
            frames = [cv2.resize(np.ascontiguousarray(f), size, interpolation=cv2.INTER_AREA) for f in frames]
        return _crop(frames, rng), cond, (has_m2, has_p2)



def _dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class VimeoStream:
    """Streams one parquet shard at a time into RAM (PNG bytes), deleting the file; cycles through all 25 shards."""

    @classmethod
    def from_items(cls, items):
        """Fixed in-RAM triplet list (e.g. a cached shard for CPU design checks); no streaming."""
        obj = cls.__new__(cls)
        obj.items, obj.lock, obj.pos = items, threading.Lock(), 1
        return obj

    def __init__(self, cache: Path, seed: int, start: int = 0):
        self.cache, self.order = cache, list(range(25))
        random.Random(seed).shuffle(self.order)
        self.pos, self.items, self.lock = start, [], threading.Lock()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.load_next()

    def load_next(self) -> None:
        import pyarrow.parquet as pq
        shard = self.order[self.pos % 25]
        tmp = self.cache / f'vimeo-{shard}.parquet.part'
        urllib.request.urlretrieve(VIMEO_REPO.format(shard), tmp)
        table = pq.read_table(tmp, columns=['im1', 'im2', 'im3'])
        items = [tuple(table.column(c)[i].as_py()['bytes'] for c in ('im1', 'im2', 'im3')) for i in range(table.num_rows)]
        del table
        tmp.unlink()
        with self.lock:
            self.items, self.pos = items, self.pos + 1
        print(f'vimeo shard {shard} loaded ({len(items)} triplets); shards seen {self.pos}', flush=True)

    def sample(self, rng: random.Random, **_):
        with self.lock:
            k = rng.randrange(len(self.items))
        return self.build(rng, k)

    def build(self, rng: random.Random, k: int):
        with self.lock:
            im1, im2, im3 = self.items[k]
        a, mid, b = _dec(im1), _dec(im2), _dec(im3)
        if rng.random() < .5:
            a, b = b, a
        cond = make_cond(.5, 1, 0., 0., False, False)
        return _crop([a, b, a, b, mid], rng), cond, (False, False)


def batch(gopro: GoproSource, vimeo: VimeoStream, size: int, rng: random.Random):
    """Returns tensors a_m1, a_p1, a_m2, a_p2, target (B,3,256,256) in [0,1], cond (B,8)."""
    frames, conds = [], []
    for _ in range(size):
        crop, cond, (has_m2, has_p2) = (gopro if rng.random() < GOPRO_P else vimeo).sample(rng)
        if not has_m2:
            crop[2] = np.zeros_like(crop[2])
        if not has_p2:
            crop[3] = np.zeros_like(crop[3])
        frames.append(np.stack([np.ascontiguousarray(c) for c in crop]))
        conds.append(cond)
    x = torch.from_numpy(np.stack(frames)).permute(1, 0, 4, 2, 3).float() / 255.   # (5, B, 3, H, W)
    return x[0], x[1], x[2], x[3], x[4], torch.tensor(conds, dtype=torch.float32)


def inner_motion(crop) -> float:
    """Mean inner-anchor displacement (full-res px) of a sample, Lucas-Kanade at 1/4 resolution."""
    return pair_motion(crop[0], crop[1])


def pair_motion(a, b) -> float:
    """Mean displacement (px) between two HxWx3 uint8 frames, Lucas-Kanade at 1/4 resolution."""
    h, w = (a.shape[0] // 4) * 4, (a.shape[1] // 4) * 4
    a = torch.from_numpy(np.ascontiguousarray(a[:h, :w], dtype=np.float32).transpose(2, 0, 1)[None] / 255.)
    b = torch.from_numpy(np.ascontiguousarray(b[:h, :w], dtype=np.float32).transpose(2, 0, 1)[None] / 255.)
    with torch.no_grad():
        return float(lk_flow(a, b, out_scale=4).norm(dim=1).mean() * 4)


def _batched_motion(pairs, chunk: int = 64) -> np.ndarray:
    """Mean displacement (px of the given frames) for a list of (a, b) uint8 frame pairs. Lucas-Kanade on
    quarter-resolution frames with three levels: the same computation as lk_flow(..., out_scale=4)."""
    out = []
    for i in range(0, len(pairs), chunk):
        ch = pairs[i:i + chunk]
        q = lambda f: cv2.resize(np.ascontiguousarray(f), (f.shape[1] // 4, f.shape[0] // 4), interpolation=cv2.INTER_AREA)
        a = torch.from_numpy(np.stack([q(x) for x, _ in ch]).astype(np.float32).transpose(0, 3, 1, 2) / 255.)
        b = torch.from_numpy(np.stack([q(y) for _, y in ch]).astype(np.float32).transpose(0, 3, 1, 2) / 255.)
        with torch.no_grad():
            out.append((lk_flow(a, b, levels=3).norm(dim=1).mean((1, 2)) * 4).numpy())
    return np.concatenate(out) if out else np.zeros(0)


class StratifiedMotion:
    """Training samples whose inner-anchor motion follows a target histogram (DESIGN_AUDIT section 7).

    GOPRO_Large is handheld deblurring footage: its anchor pairs move 4x more than the dev missing-frame
    brackets (median 49 vs 12 px) and never below 4 px. Training on it as drawn teaches the model to
    hedge, i.e. blur. Here every GOPRO anchor pair (gap <= 16) and every Vimeo triplet gets its exact
    motion m once (batched Lucas-Kanade). A sample is drawn by choosing a motion bin from the target, then
    a pair that can land in it; GOPRO pairs are resized by a factor s in [min_scale, 1] chosen to put s*m
    inside the bin (displacement scales exactly with resolution). Exact matching, no rejection.
    Target = dev bracket histogram (statistics only; no dev frames are trained on)."""

    def __init__(self, gopro, vimeo, target_hist, bins, min_scale: float = 0.72, cache: Path | None = None):
        self.gopro, self.vimeo, self.min_scale = gopro, vimeo, min_scale
        self.bins = np.asarray(bins, dtype=float)
        self.target = np.asarray(target_hist, dtype=float)
        self.g_index = self._gopro_table(cache)
        self._vimeo_gen = None
        self._refresh_vimeo()

    def _gopro_table(self, cache):
        if cache is not None and Path(cache).exists():
            return np.load(cache)
        rows, pairs = [], []
        for si, seq in enumerate(self.gopro.scenes):
            n = len(seq)
            for g in range(1, min(16, n - 3) + 1):
                for left in range(0, n - g - 1):
                    rows.append((si, left, g)); pairs.append((seq[left], seq[left + g + 1]))
        m = _batched_motion(pairs)
        table = np.concatenate([np.asarray(rows, dtype=np.float64), m[:, None]], 1)   # si, left, g, motion
        if cache is not None:
            np.save(cache, table)
        return table

    def _refresh_vimeo(self):
        if self.vimeo is None:
            if self._vimeo_gen is None:
                self.v_motion, self._vimeo_gen = np.zeros(0), -1
                self._plan()
            return
        if self._vimeo_gen == self.vimeo.pos:
            return
        with self.vimeo.lock:
            items = list(self.vimeo.items)
        self.v_motion = _batched_motion([(_dec(a), _dec(b)) for a, _, b in items])
        self._vimeo_gen = self.vimeo.pos
        self._plan()

    def _plan(self):
        """Per bin: candidate GOPRO rows (resizable into the bin) and Vimeo items (already inside it)."""
        gm, lo_s = self.g_index[:, 3], self.min_scale
        self.cand = []
        for k in range(len(self.target)):
            lo, hi = self.bins[k], self.bins[k + 1]
            g = np.nonzero((gm * lo_s < hi) & (gm >= lo))[0]
            v = np.nonzero((self.v_motion >= lo) & (self.v_motion < hi))[0] if self.vimeo is not None else np.zeros(0, int)
            self.cand.append((g, v))
        reach = np.array([len(g) + len(v) > 0 for g, v in self.cand])
        self.coverage = float(self.target[reach].sum() / self.target.sum())
        self.p = self.target * reach / (self.target * reach).sum()

    def sample(self, rng: random.Random):
        self._refresh_vimeo()
        k = int(np.searchsorted(np.cumsum(self.p), rng.random() * self.p.sum(), side='right'))
        k = min(k, len(self.p) - 1)
        g, v = self.cand[k]
        if rng.random() < len(g) / (len(g) + len(v)):
            si, left, gap, m = self.g_index[g[rng.randrange(len(g))]]
            lo, hi = self.bins[k], self.bins[k + 1]
            floor = max(self.min_scale, (CROP + 1) / min(self.gopro.scenes[int(si)][0].shape[:2]))
            s_lo, s_hi = max(floor, lo / max(m, 1e-6)), min(1.0, hi / max(m, 1e-6))
            sc = rng.uniform(s_lo, s_hi) if s_hi > s_lo else 1.0
            return self.gopro.build(rng, int(si), int(left), int(gap), sc)
        return self.vimeo.build(rng, int(v[rng.randrange(len(v))]))

    def summary(self):
        return dict(coverage=round(self.coverage, 3), target=self.target.round(4).tolist(),
                    candidates=[[int(len(g)), int(len(v))] for g, v in self.cand], gopro_pairs=int(len(self.g_index)))
