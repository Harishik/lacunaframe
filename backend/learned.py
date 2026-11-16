"""Pinned pretrained RIFE inference, CPU only. No reference pixels or network calls."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

ASSETS=Path(__file__).resolve().parents[1]/'assets'
MANIFEST=json.loads((ASSETS/'manifest.json').read_text())
THREADS=max(1,min(4,int(os.environ.get('LACUNA_CPU_THREADS','2'))))

@lru_cache(maxsize=1)
def model_info() -> dict:
    info=MANIFEST['model']
    p=ASSETS/info['file']
    available=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
    return dict(**info,available=available,provider='CPUExecutionProvider',threads=THREADS,runtime=ort.__version__,optimization='ORT_ENABLE_ALL',precision='float32')

@lru_cache(maxsize=1)
def session() -> ort.InferenceSession:
    if not model_info()['available']:
        raise RuntimeError('Pinned RIFE weights missing or checksum mismatch. Run python scripts/fetch_assets.py.')
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=THREADS
    opts.inter_op_num_threads=1
    opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(ASSETS/MANIFEST['model']['file']),sess_options=opts,providers=['CPUExecutionProvider'])


def interpolate(a: np.ndarray,b: np.ndarray,t: float,fast: bool=False) -> np.ndarray:
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3 or not 0<=t<=1:
        raise ValueError('RIFE requires matching RGB images and timestep in [0,1]')
    if t==0:return a.copy()
    if t==1:return b.copy()
    h,w=a.shape[:2]
    if fast:
        a=cv2.resize(a,(max(8,w//2),max(8,h//2)),interpolation=cv2.INTER_AREA)
        b=cv2.resize(b,(max(8,w//2),max(8,h//2)),interpolation=cv2.INTER_AREA)
    hh,ww=a.shape[:2]
    def tensor(frame: np.ndarray) -> np.ndarray:
        rgb=frame.astype(np.float32).transpose(2,0,1)[None]/255.
        return np.pad(rgb,((0,0),(0,0),(0,(-hh)%32),(0,(-ww)%32)),mode='constant')
    out=session().run(['output'],{'img0':tensor(a),'img1':tensor(b),'timestep':np.array([t],np.float32)})[0]
    if not np.isfinite(out).all():raise RuntimeError('Nonfinite RIFE output')
    result=out[0,:,:hh,:ww].transpose(1,2,0)
    if fast:result=cv2.resize(result,(w,h),interpolation=cv2.INTER_LINEAR)
    return np.clip(result*255,0,255).round().astype(np.uint8)
