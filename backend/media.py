"""Frame normalisation, upload decoding and preview encoding."""
from __future__ import annotations
import io
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np
from . import decode

WIDTH, HEIGHT = 320, 180
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_PIXELS = 1920 * 1080
MAX_SECONDS = 8.1
MAX_SAMPLES = 96
UPLOAD_FPS = 12.


def normalize(a: np.ndarray) -> np.ndarray:
    """Letterbox any frame into a WIDTH x HEIGHT canvas, preserving aspect ratio (channel order unchanged)."""
    h, w = a.shape[:2]
    scale = min(WIDTH / w, HEIGHT / h)
    small = cv2.resize(a, (max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    hh, ww = small.shape[:2]
    canvas[(HEIGHT - hh) // 2:(HEIGHT - hh) // 2 + hh, (WIDTH - ww) // 2:(WIDTH - ww) // 2 + ww] = small
    return canvas


def encode(folder: Path, name: str, frames: np.ndarray, fps: float) -> str:
    """Write an H.264 MP4 preview, falling back to GIF if the encoder is unavailable."""
    try:
        imageio.mimwrite(folder / f'{name}.mp4', frames, fps=fps, codec='libx264', macro_block_size=1, pixelformat='yuv420p')
        if (folder / f'{name}.mp4').stat().st_size < 100:
            raise RuntimeError('Empty encoder output')
        return f'{name}.mp4'
    except Exception:
        (folder / f'{name}.mp4').unlink(missing_ok=True)
        imageio.mimsave(folder / f'{name}.gif', frames, duration=1000 / fps, loop=0)
        return f'{name}.gif'


def decode_image(data: bytes, previous_shape: tuple | None) -> np.ndarray:
    """Decode one PNG/JPEG (BGR, normalised). Header is checked before pixels are allocated."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        if im.width * im.height > MAX_PIXELS:
            raise ValueError('Image resolution exceeds 1920×1080.')
    a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if a is None:
        raise ValueError('Image decode failed.')
    if previous_shape is not None and previous_shape != a.shape:
        raise ValueError('Sequence images must have matching dimensions.')
    return a


def decode_video(raw: Path) -> tuple[list[np.ndarray], float]:
    """Decode an uploaded video at <= UPLOAD_FPS; returns normalised BGR frames and the sampling rate."""
    limits = 'Video must be decodable, ≤8 seconds and ≤1920×1080.'
    try:
        info = decode.probe(raw)
    except Exception as exc:
        raise ValueError(limits) from exc
    fps, count = info.fps, info.frame_count
    if not np.isfinite(fps) or fps <= 0 or count < 2 or count / fps > MAX_SECONDS or info.width * info.height > MAX_PIXELS:
        raise ValueError(limits)
    target, next_time, index, decoded = min(UPLOAD_FPS, fps), 0., 0, []
    try:
        for a in decode.frames(raw):
            if index > 2000:
                raise ValueError('Input frame limit exceeded.')
            if index / fps + 1e-6 >= next_time:
                # Uploaded images arrive as BGR from cv2.imdecode; keep one channel order.
                decoded.append(normalize(cv2.cvtColor(a, cv2.COLOR_RGB2BGR)))
                next_time += 1 / target
            index += 1
            if len(decoded) > MAX_SAMPLES:
                raise ValueError('More than 96 sampled frames.')
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError('Video is truncated or partially undecodable.') from exc
    if np.isfinite(count) and index < count - 1:
        raise ValueError('Video is truncated or partially undecodable.')
    return decoded, target
