"""LPIPS (AlexNet, v0.1) and DISTS on CPU through ONNX Runtime.

The graphs are exported once from the reference PyTorch implementations by
research/metrics/export_perceptual.py and validated against float64 PyTorch
(max error 4e-8 LPIPS, 3.5e-7 DISTS). They live outside git in
$LACUNA_EXTERNAL/weights; when absent, `available()` is False and callers skip
perceptual metrics instead of failing.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import numpy as np

WEIGHTS = Path(os.environ.get('LACUNA_EXTERNAL', Path(__file__).resolve().parents[2] / 'lacuna-data')) / 'weights'
FILES = {'lpips': 'lpips_alex_v0.1.onnx', 'dists': 'dists.onnx'}
MANIFEST_KEYS = {'lpips': 'lpips_alex', 'dists': 'dists'}


def available(metric: str) -> bool:
    return (WEIGHTS / FILES[metric]).is_file()


@lru_cache(maxsize=2)
def _session(metric: str):
    import onnxruntime as ort
    path = WEIGHTS / FILES[metric]
    manifest = json.loads((WEIGHTS / 'perceptual-manifest.json').read_text())
    expected = manifest[MANIFEST_KEYS[metric]]['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f'{metric} weights checksum mismatch')
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get('LACUNA_CPU_THREADS', '2'))
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def _tensor(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 3:
        frames = frames[None]
    if frames.dtype != np.uint8 or frames.shape[-1] != 3:
        raise ValueError('RGB uint8 NHWC frames required')
    return np.ascontiguousarray(frames.transpose(0, 3, 1, 2), dtype=np.float32) / 255.


def distance(metric: str, ref: np.ndarray, out: np.ndarray, batch: int = 4) -> np.ndarray:
    """Per-frame perceptual distance (lower is better) between aligned RGB uint8 frames."""
    if ref.shape != out.shape:
        raise ValueError('Aligned frames required')
    a_all, b_all = _tensor(ref), _tensor(out)
    session = _session(metric)
    values = [session.run(None, {'a': a_all[i:i + batch], 'b': b_all[i:i + batch]})[0] for i in range(0, len(a_all), batch)]
    return np.concatenate(values).astype(float)
