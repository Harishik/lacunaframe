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
from .model import make_cond

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
    def __init__(self, folder: Path):
        self.scenes = [np.load(p)['frames'] for p in sorted(folder.glob('*.npz'))]
        if not self.scenes:
            raise FileNotFoundError(f'No GOPRO training scenes in {folder}')

    def sample(self, rng: random.Random):
        seq = rng.choice(self.scenes)
        n = len(seq)
        g = rng.randint(1, min(16, n - 3))
        left = rng.randint(0, n - g - 2)
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
        return _crop(frames, rng), cond, (has_m2, has_p2)


class VimeoStream:
    """Streams one parquet shard at a time into RAM (PNG bytes), deleting the file; cycles through all 25 shards."""

    def __init__(self, cache: Path, seed: int):
        self.cache, self.order = cache, list(range(25))
        random.Random(seed).shuffle(self.order)
        self.pos, self.items, self.lock = 0, [], threading.Lock()
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

    def sample(self, rng: random.Random):
        with self.lock:
            im1, im2, im3 = rng.choice(self.items)
        dec = lambda b: cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        a, mid, b = dec(im1), dec(im2), dec(im3)
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
