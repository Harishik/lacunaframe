"""CPU ONNX adapters for published interpolators exported by research/metrics/export_baselines.py.

Graph signature: img0, img1 (RGB float32 NCHW in [0, 1], already padded) and
embt (float32 [1,1,1,1] timestep; pruned from midpoint-only graphs). Graphs are
exported at fixed padded shapes because the official code builds sampling grids
from tensor sizes, so one file exists per padded resolution.

Padding replicates each model's official inference code:
  AMT-S, IFRNet(-S): centred replicate padding to a multiple of 16 (InputPadder)
  RIFE paper (ECCV 2022): zero padding on the right/bottom to a multiple of 32
Every export passed gate P1 (ONNX vs PyTorch max abs error < 1e-3 on 8 real pairs;
recorded in $LACUNA_EXTERNAL/weights/baselines-manifest.json, which also pins hashes).
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import numpy as np

WEIGHTS = Path(os.environ.get('LACUNA_EXTERNAL', Path(__file__).resolve().parents[2] / 'lacuna-data')) / 'weights'
MIDPOINT_ONLY = {'rife_paper'}


@lru_cache(maxsize=1)
def manifest() -> dict:
    return json.loads((WEIGHTS / 'baselines-manifest.json').read_text())


def available(name: str) -> bool:
    try:
        return name in manifest()
    except FileNotFoundError:
        return False


def _padding(name: str, h: int, w: int) -> tuple[int, int, int, int]:
    """(top, bottom, left, right) padding for an h x w frame."""
    entry = manifest()[name]
    d = entry['divisor']
    ph, pw = (-h) % d, (-w) % d
    if entry['padding'] == 'replicate_center':
        return ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
    return 0, ph, 0, pw


@lru_cache(maxsize=8)
def _session(name: str, shape: str):
    import onnxruntime as ort
    files = manifest()[name]['files']
    if shape not in files:
        raise RuntimeError(f'No {name} export for padded shape {shape}; available: {sorted(files)}')
    info = files[shape]
    path = WEIGHTS / info['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256'] or not info['p1_passed']:
        raise RuntimeError(f'{name} {shape}: checksum mismatch or failed export parity')
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('LACUNA_CPU_THREADS', '2'))
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def interpolate(name: str, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 3 or not 0 <= t <= 1:
        raise ValueError('Matching RGB frames and t in [0, 1] required')
    if t == 0:
        return a.copy()
    if t == 1:
        return b.copy()
    if name in MIDPOINT_ONLY and abs(t - .5) > 1e-9:
        raise ValueError(f'{name} is a midpoint-only model (its network ignores the timestep)')
    h, w = a.shape[:2]
    top, bottom, left, right = _padding(name, h, w)
    mode = 'edge' if manifest()[name]['padding'] == 'replicate_center' else 'constant'

    def tensor(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.
        return np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)), mode=mode)

    session = _session(name, f'{h + top + bottom}x{w + left + right}')
    feeds = {'img0': tensor(a), 'img1': tensor(b), 'embt': np.full((1, 1, 1, 1), t, np.float32)}
    out = session.run(None, {i.name: feeds[i.name] for i in session.get_inputs()})[0]
    if not np.isfinite(out).all():
        raise RuntimeError(f'Nonfinite {name} output')
    crop = out[0, :, top:top + h, left:left + w].transpose(1, 2, 0)
    return np.clip(crop * 255, 0, 255).round().astype(np.uint8)
