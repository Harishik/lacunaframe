"""FastAPI routes for the LACUNAFRAME workbench.

Module layout: storage (experiment folders, jobs), media (normalise, decode
uploads, encode previews), pipeline (background reconstruction), core
(corruption, solvers, evaluation). Names used by scripts and tests are
re-exported here for compatibility.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import cv2
import imageio.v2 as imageio  # noqa: F401  (re-exported; tests patch imageio.mimwrite)
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from . import decode, storage
from .core import corrupt, demo
from .learned import ASSETS, MANIFEST, model_info
from .media import MAX_UPLOAD_BYTES, UPLOAD_FPS, decode_image, decode_video, encode, normalize
from .pipeline import process
from .storage import NATURAL_IDS, ROOT, cleanup, create, jobs, locate, meta, save_meta  # noqa: F401

logging.basicConfig(level=logging.INFO, format='%(message)s')
executor = ThreadPoolExecutor(max_workers=1)
PUBLIC_METHODS = ('hold', 'linear', 'flow', 'telea', 'rife', 'rife_fast')
VIEWS = ('original', 'corrupted', *PUBLIC_METHODS, 'mask', 'error')
DOWNLOADS = {'report.json', 'metrics.csv'} | {f'{m}.{ext}' for m in (*PUBLIC_METHODS, 'original', 'corrupted') for ext in ('mp4', 'gif')}


def _prepare_demos() -> None:
    for i, name in enumerate(storage.DEMO_NAMES):
        p = storage.ROOT / f'demo{i}'
        p.mkdir(exist_ok=True)
        if not (p / 'reference.npy').exists():
            frames = demo(i)
            np.save(p / 'reference.npy', frames)
            encode(p, 'original', frames, 12)
        save_meta(p, dict(id=f'demo{i}', name=name, fps=12, count=48, width=320, height=180, synthetic=True))


def _prepare_natural() -> None:
    for video in MANIFEST['videos']:
        source = ASSETS / video['file']
        if not source.exists():
            continue
        if hashlib.sha256(source.read_bytes()).hexdigest() != video['sha256']:
            raise RuntimeError('Natural-video checksum mismatch: ' + video['file'])
        p = storage.ROOT / video['id']
        p.mkdir(exist_ok=True)
        # Deterministic FFmpeg decode: identical references on every OS (see backend/decode.py).
        source_fps = decode.probe(source).fps
        decoded_rgb, indices = decode.sampled(source, every=2, limit=48)
        if len(decoded_rgb) < 2:
            raise RuntimeError('Natural clip failed to decode')
        ref = np.stack([normalize(a) for a in decoded_rgb])
        np.save(p / 'reference.npy', ref)
        if not (p / 'original.mp4').exists():
            encode(p, 'original', ref, source_fps / 2)
        save_meta(p, dict(**video, fps=source_fps / 2, count=len(ref), width=320, height=180, synthetic=False,
                          source_fps=source_fps, source_indices=indices, source_timestamps=[i / source_fps for i in indices],
                          reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                          reference_note='Decoded MP4 mirror, letterboxed and sampled every second frame; not camera RAW.'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _prepare_demos()
    _prepare_natural()
    storage.stop.clear()
    thread = threading.Thread(target=storage.janitor, daemon=True)
    thread.start()
    yield
    storage.stop.set()
    thread.join(timeout=2)


app = FastAPI(title='LACUNAFRAME', version='0.3.0', lifespan=lifespan)


class Scenario(BaseModel):
    source: str
    seed: int = Field(42, ge=0, le=2**32 - 1)
    drop: float = Field(.15, ge=0, le=.8)
    gap: int = Field(3, ge=0, le=48)
    start: float = Field(.4, ge=0, le=1)
    occlusion: float = Field(0, ge=0, le=.7)
    mask_shape: Literal['rectangle', 'brush'] = 'rectangle'
    noise: float = Field(0, ge=0, le=60)
    blur: float = Field(0, ge=0, le=3)


class Run(BaseModel):
    scenario: str
    methods: list[Literal['hold', 'linear', 'flow', 'telea', 'rife', 'rife_fast']] = Field(default=['hold', 'linear', 'flow'], min_length=1, max_length=6)


@app.get('/api/demos')
def demos() -> list[dict]:
    return [meta(i) for i in NATURAL_IDS if (storage.ROOT / i).exists()] + [meta(i) for i in storage.DEMO_IDS]


@app.get('/api/benchmarks')
def benchmarks() -> dict:
    path = Path(__file__).resolve().parents[1] / 'examples/natural-v02/summary.json'
    if not path.exists():
        return dict(summary=[], protocol='Natural-video benchmark has not been generated yet.')
    report = json.loads(path.read_text())
    return dict(summary=report['summary'], protocol=report['protocol'])


@app.get('/api/capabilities')
def capabilities() -> dict:
    return dict(model=model_info(), natural_sources=MANIFEST['videos'])


async def _read_limited(f: UploadFile, total: int) -> tuple[bytearray, int]:
    data = bytearray()
    while chunk := await f.read(1024 * 1024):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise ValueError('Upload exceeds 30 MiB total.')
        data.extend(chunk)
    return data, total


@app.post('/api/upload')
async def upload(files: list[UploadFile] = File(...)) -> dict:
    if not 1 <= len(files) <= 96:
        raise HTTPException(400, 'Choose one MP4/MOV or 2–96 PNG/JPEG images (lexical filename order, 12 fps).')
    ident, p = create()
    total = 0
    try:
        images = len(files) > 1 or files[0].content_type in ('image/png', 'image/jpeg')
        allowed = ('image/png', 'image/jpeg') if images else ('video/mp4', 'video/quicktime')
        decoded, shape, fps = [], None, UPLOAD_FPS
        for f in sorted(files, key=lambda f: f.filename or ''):
            if f.content_type not in allowed:
                raise ValueError('Unsupported MIME type. Use MP4/MOV or PNG/JPEG sequence.')
            data, total = await _read_limited(f, total)
            if images:
                a = decode_image(bytes(data), shape)
                shape = a.shape
                decoded.append(normalize(a))
            else:
                # FFmpeg selects the demuxer from the suffix; derive it from the validated MIME type, never the client filename.
                raw = p / ('input.mov' if f.content_type == 'video/quicktime' else 'input.mp4')
                raw.write_bytes(data)
                try:
                    decoded, fps = decode_video(raw)
                finally:
                    raw.unlink(missing_ok=True)
        if len(decoded) < 2:
            raise ValueError('At least two decoded frames required.')
        ref = np.stack([cv2.cvtColor(normalize(a), cv2.COLOR_BGR2RGB) for a in decoded])
        np.save(p / 'reference.npy', ref)
        info = dict(id=ident, name='Local upload', fps=fps, count=len(ref), width=320, height=180, synthetic=False,
                    sha256=hashlib.sha256(ref.tobytes()).hexdigest())
        save_meta(p, info)
        return info
    except Exception as exc:
        shutil.rmtree(p, ignore_errors=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(400, str(exc)) from exc
    finally:
        for f in files:
            await f.close()


@app.post('/api/scenarios')
def scenario(config: Scenario) -> dict:
    source = locate(config.source)
    ref = np.load(source / 'reference.npy')
    ident, p = create()
    damaged, missing, mask = corrupt(ref, config.model_dump())
    for name, array in [('reference', ref), ('corrupted', damaged), ('missing', missing), ('mask', mask)]:
        np.save(p / f'{name}.npy', array)
    info = {**meta(config.source), 'id': ident, 'config': config.model_dump(), 'missing': missing.tolist(),
            'sha256': hashlib.sha256(ref.tobytes()).hexdigest()}
    save_meta(p, info)
    return info


@app.post('/api/run')
def run(request: Run) -> dict:
    source = locate(request.scenario)
    if not (source / 'corrupted.npy').exists():
        raise HTTPException(400, 'Create a scenario first')
    if any(m.startswith('rife') for m in request.methods) and not model_info()['available']:
        raise HTTPException(503, 'RIFE weights unavailable; run python scripts/fetch_assets.py')
    if not storage.gate.acquire(blocking=False):
        raise HTTPException(429, 'CPU worker busy. Wait for the current run.')
    try:
        ident, p = create()
        save_meta(p, {**meta(request.scenario), 'id': ident, 'scenario': request.scenario})
        # Self-contained run: survives source expiration.
        for name in ('reference', 'corrupted', 'missing', 'mask'):
            shutil.copyfile(source / f'{name}.npy', p / f'{name}.npy')
        jobs[ident] = dict(id=ident, status='running', progress=0)
        executor.submit(process, ident, p, list(dict.fromkeys(request.methods)))
        return jobs[ident]
    except Exception:
        storage.gate.release()
        raise


@app.get('/api/results/{ident}')
def results(ident: str) -> dict:
    p = locate(ident)
    if (p / 'report.json').exists():
        return dict(id=ident, status='complete', progress=1, report=json.loads((p / 'report.json').read_text()))
    return jobs.get(ident, dict(id=ident, status='failed', error='Run interrupted by server restart'))


def _missing_placeholder(a: np.ndarray) -> None:
    a[:] = (35, 28, 30)
    cv2.putText(a, 'MISSING SAMPLE', (43, 94), cv2.FONT_HERSHEY_SIMPLEX, .6, (120, 140, 250), 1, cv2.LINE_AA)


@app.get('/api/frames/{ident}/{kind}/{index}')
def frame(ident: str, kind: str, index: int, method: str = 'flow') -> Response:
    p = locate(ident)
    if kind not in VIEWS:
        raise HTTPException(404, 'Unknown view')
    name = 'reference' if kind == 'original' else kind
    if kind == 'error':
        if method not in PUBLIC_METHODS:
            raise HTTPException(404, 'Unknown method')
        name = method
    path = p / f'{name}.npy'
    if not path.exists():
        raise HTTPException(404, 'View not available yet')
    arr = np.load(path, mmap_mode='r')
    if not 0 <= index < len(arr):
        raise HTTPException(404, 'Frame out of bounds')
    a = arr[index].copy()
    if kind == 'error':
        ref = np.load(p / 'reference.npy', mmap_mode='r')[index]
        a = cv2.applyColorMap(np.abs(a.astype(float) - ref.astype(float)).mean(-1).astype('uint8'), cv2.COLORMAP_INFERNO)
    elif kind == 'mask':
        if np.load(p / 'missing.npy')[index]:
            a[:] = True
        a = np.repeat((a.astype('uint8') * 255)[..., None], 3, -1)
    else:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        if kind == 'corrupted' and np.load(p / 'missing.npy')[index]:
            _missing_placeholder(a)
    ok, data = cv2.imencode('.png', a)
    return Response(data.tobytes(), media_type='image/png', headers={'Cache-Control': 'private, max-age=300'})


@app.get('/api/download/{ident}/{filename}')
def download(ident: str, filename: str) -> FileResponse:
    if filename not in DOWNLOADS:
        raise HTTPException(404, 'Unknown artifact')
    p = locate(ident) / filename
    if not p.exists():
        raise HTTPException(404, 'Artifact unavailable')
    return FileResponse(p, filename=filename)


# Optional single-container mode (e.g. a Hugging Face Space): serve the built
# frontend from the same origin. Mounted last so every /api route takes precedence.
if os.environ.get('LACUNA_STATIC'):
    from fastapi.staticfiles import StaticFiles
    app.mount('/', StaticFiles(directory=os.environ['LACUNA_STATIC'], html=True), name='frontend')
