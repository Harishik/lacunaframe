"""Deterministic video decoding through the pinned imageio-ffmpeg binary.

OpenCV picks a platform decoder at runtime: FFmpeg on Linux wheels, but
AVFoundation on macOS wheels. On the same H.264 file those backends disagree by
several 8-bit levels per pixel (up to 71 on DAVIS bmx-trees), which silently
changes every reference frame and therefore every score. All evidence paths
decode through this module so references are identical on every host. FFmpeg
decoding reproduces the archived Linux results bit-exactly.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import math
import imageio.v2 as imageio
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: float  # may be math.inf when the container does not record it


def probe(path: str | Path) -> VideoInfo:
    """Read container metadata without decoding every frame."""
    with imageio.get_reader(str(path), 'ffmpeg') as reader:
        meta = reader.get_meta_data()
    width, height = meta.get('size', (0, 0))
    fps = float(meta.get('fps') or 0.0)
    duration = float(meta.get('duration') or 0.0)
    count = meta.get('nframes', math.inf)
    if not isinstance(count, (int, float)) or not math.isfinite(count):
        count = round(duration * fps) if duration and fps else math.inf
    return VideoInfo(fps=fps, width=int(width), height=int(height), frame_count=float(count))


def frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield RGB uint8 HxWx3 frames in decode order."""
    with imageio.get_reader(str(path), 'ffmpeg') as reader:
        for frame in reader:
            yield np.ascontiguousarray(frame[..., :3], dtype=np.uint8)


def sampled(path: str | Path, every: int = 2, limit: int = 48) -> tuple[list[np.ndarray], list[int]]:
    """Keep every `every`-th decoded RGB frame, up to `limit` frames, with source indices."""
    kept: list[np.ndarray] = []
    indices: list[int] = []
    for index, frame in enumerate(frames(path)):
        if len(kept) >= limit:
            break
        if index % every == 0:
            kept.append(frame)
            indices.append(index)
    return kept, indices


def count(path: str | Path) -> int:
    """Number of decodable frames (full decode)."""
    return sum(1 for _ in frames(path))
