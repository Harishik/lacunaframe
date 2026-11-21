"""CPU ONNX runtime for LacunaNet (our model; protocol v4).

The exported graph takes inner anchors, optional outer anchors and an 8-value
conditioning vector, and returns the reconstruction plus a per-pixel Laplace
scale (expected |error| / 255 units in [0, 1] space). Weights live outside git
in $LACUNA_EXTERNAL/models/lacunanet/; `LACUNANET_ONNX` can point to a specific
export (used for checkpoint selection).
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get('LACUNA_EXTERNAL', Path(__file__).resolve().parents[2] / 'lacuna-data')) / 'models' / 'lacunanet'
DIVISOR = 16   # v2 has a 1/16 pyramid level; also valid for v1 (multiple of 8)


def model_path() -> Path:
    return Path(os.environ.get('LACUNANET_ONNX', ROOT / 'lacunanet.onnx'))


def available() -> bool:
    return model_path().is_file()


@lru_cache(maxsize=4)
def _session(path: str):
    import onnxruntime as ort
    manifest = Path(path).with_suffix('.json')
    if manifest.exists():
        expected = json.loads(manifest.read_text())['sha256']
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError('LacunaNet weights checksum mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('LACUNA_CPU_THREADS', '2'))
    options.inter_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])


def cond_vector(t: float, gap: int, left: int, right: int, left2: int | None, right2: int | None) -> np.ndarray:
    """Same layout as research.lacunanet.model.make_cond, in units of the inner span."""
    span = right - left
    clip = lambda v: max(-4.0, min(4.0, v))
    tau_m2 = clip(-t - (left - left2) / span) if left2 is not None else 0.0
    tau_p2 = clip(1 - t + (right2 - right) / span) if right2 is not None else 0.0
    return np.array([[t, math.log2(max(gap, 1)) / 5.0, tau_m2, -t, 1 - t, tau_p2,
                      float(left2 is not None), float(right2 is not None)]], np.float32)


def reconstruct(frames: np.ndarray, left: int, right: int, left2: int | None, right2: int | None, index: int):
    """Reconstruct frame `index` from observed anchors. Returns (uint8 HxWx3, scale HxW float32 in 0..255 units)."""
    h, w = frames.shape[1:3]
    ph, pw = (-h) % DIVISOR, (-w) % DIVISOR

    def tensor(i: int | None) -> np.ndarray:
        if i is None:
            return np.zeros((1, 3, h + ph, w + pw), np.float32)
        x = frames[i].astype(np.float32).transpose(2, 0, 1)[None] / 255.
        return np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode='edge')

    gap = right - left - 1
    t = (index - left) / (right - left)
    feeds = dict(a_m1=tensor(left), a_p1=tensor(right), a_m2=tensor(left2), a_p2=tensor(right2),
                 cond=cond_vector(t, gap, left, right, left2, right2))
    frame, scale = _session(str(model_path())).run(None, feeds)
    if not np.isfinite(frame).all():
        raise RuntimeError('Nonfinite LacunaNet output')
    out = np.clip(frame[0, :, :h, :w].transpose(1, 2, 0) * 255, 0, 255).round().astype(np.uint8)
    return out, scale[0, 0, :h, :w] * 255.0
