"""Opt-in RIFE 4.26 research candidate; CPU and checksum pinned.

vs-mlrt v1 ABI: RGB A, RGB B, t, normalized x/y grid, 2/(W-1),
2/(H-1). Coordinates use the padded dimensions (multiple of 64).
Reference: AmusementClub/vs-mlrt scripts/vsmlrt.py:get_rife_input/RIFEMerge.
No weights are trained, pruned, distilled or modified here.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from .learned import ASSETS, THREADS

INFO = json.loads((ASSETS/'candidate.json').read_text())

@lru_cache(maxsize=1)
def session() -> ort.InferenceSession:
    path = ASSETS/INFO['file']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != INFO['sha256']:
        raise RuntimeError('Candidate weights unavailable or invalid; run scripts/fetch_assets.py --candidate')
    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def pack(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 2 or not 0 <= t <= 1:
        raise ValueError('Matching RGB frames >=2 pixels and timestep in [0,1] required')
    h,w = a.shape[:2]
    def tensor(frame: np.ndarray) -> np.ndarray:
        return np.pad(frame.astype(np.float32).transpose(2,0,1)/255,
                      ((0,0),(0,(-h)%64),(0,(-w)%64)))
    first,second = tensor(a),tensor(b)
    H,W = first.shape[1:]
    x,y = np.meshgrid(np.linspace(-1,1,W,dtype=np.float32),np.linspace(-1,1,H,dtype=np.float32))
    constant = lambda v: np.full((H,W),v,np.float32)
    aux = np.stack([constant(t),x,y,constant(2/(W-1)),constant(2/(H-1))])
    return np.concatenate([first,second,aux])[None]


def interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    inputs = pack(a,b,t)
    if t == 0: return a.copy()
    if t == 1: return b.copy()
    result = session().run(['output'], {'input':inputs})[0]
    if not np.isfinite(result).all(): raise RuntimeError('Nonfinite candidate output')
    h,w = a.shape[:2]
    return np.clip(result[0,:,:h,:w].transpose(1,2,0)*255,0,255).round().astype(np.uint8)
