from __future__ import annotations
import csv
import hashlib
import io
import json
import logging
import os
import platform
import importlib.metadata
from pathlib import Path
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal
import cv2
import imageio.v2 as imageio
import numpy as np
import psutil
from .learned import ASSETS, MANIFEST, model_info, session
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel, Field
from .core import demo, corrupt, restore, evaluate
from . import decode

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('lacunaframe')
ROOT = Path(os.environ.get('LACUNA_DATA', '/tmp/lacunaframe'))
ROOT.mkdir(parents=True,exist_ok=True)
TTL = 3600
executor = ThreadPoolExecutor(max_workers=1)
gate = threading.Lock()
stop = threading.Event()
jobs: dict[str, dict] = {}
NATURAL_IDS = [v['id'] for v in MANIFEST['videos']]
DEMO_NAMES = ['Orbital study', 'Crossing paths', 'Drifting textures']


def save_meta(folder: Path, data: dict) -> None:
    (folder/'meta.json').write_text(json.dumps(data))


def create() -> tuple[str,Path]:
    cleanup()
    if len(list(ROOT.iterdir())) >= 32:
        raise HTTPException(429,'Session storage full. Restart with a clean data directory or wait one hour.')
    ident = uuid.uuid4().hex
    folder = ROOT/ident
    folder.mkdir()
    return ident,folder


def cleanup() -> None:
    for p in ROOT.iterdir():
        if p.is_dir() and not p.name.startswith('demo') and p.name not in NATURAL_IDS and time.time()-p.stat().st_mtime>TTL:
            if jobs.get(p.name,{}).get('status') != 'running':
                shutil.rmtree(p,ignore_errors=True)
                jobs.pop(p.name,None)


def janitor() -> None:
    while not stop.wait(60):
        cleanup()


def locate(ident: str) -> Path:
    if not (ident in NATURAL_IDS+[f'demo{i}' for i in range(3)] or (len(ident)==32 and all(c in '0123456789abcdef' for c in ident))):
        raise HTTPException(404,'Unknown experiment')
    p = ROOT/ident
    if not p.is_dir():
        raise HTTPException(404,'Experiment expired or does not exist')
    return p


def meta(ident: str) -> dict:
    return json.loads((locate(ident)/'meta.json').read_text())


def normalize(a: np.ndarray) -> np.ndarray:
    h,w=a.shape[:2]; scale=min(320/w,180/h)
    small=cv2.resize(a,(max(1,round(w*scale)),max(1,round(h*scale))))
    canvas=np.zeros((180,320,3),np.uint8); hh,ww=small.shape[:2]
    canvas[(180-hh)//2:(180-hh)//2+hh,(320-ww)//2:(320-ww)//2+ww]=small
    return canvas


def encode(folder: Path, name: str, frames: np.ndarray, fps: float) -> str:
    try:
        imageio.mimwrite(folder/f'{name}.mp4',frames,fps=fps,codec='libx264',macro_block_size=1,pixelformat='yuv420p')
        if (folder/f'{name}.mp4').stat().st_size < 100:
            raise RuntimeError('Empty encoder output')
        return f'{name}.mp4'
    except Exception:
        (folder/f'{name}.mp4').unlink(missing_ok=True)
        imageio.mimsave(folder/f'{name}.gif',frames,duration=1000/fps,loop=0)
        return f'{name}.gif'


@asynccontextmanager
async def lifespan(app: FastAPI):
    for i,name in enumerate(DEMO_NAMES):
        p = ROOT/f'demo{i}'; p.mkdir(exist_ok=True)
        if not (p/'reference.npy').exists():
            frames = demo(i); np.save(p/'reference.npy',frames)
            encode(p,'original',frames,12)
        save_meta(p,dict(id=f'demo{i}',name=name,fps=12,count=48,width=320,height=180,synthetic=True))
    for video in MANIFEST['videos']:
        source=ASSETS/video['file']
        if not source.exists(): continue
        if hashlib.sha256(source.read_bytes()).hexdigest()!=video['sha256']:
            raise RuntimeError('Natural-video checksum mismatch: '+video['file'])
        p=ROOT/video['id'];p.mkdir(exist_ok=True)
        # Deterministic FFmpeg decode: identical references on every OS (see backend/decode.py).
        source_fps=decode.probe(source).fps
        decoded_rgb,indices=decode.sampled(source,every=2,limit=48)
        frames=[normalize(a) for a in decoded_rgb]
        if len(frames)<2:raise RuntimeError('Natural clip failed to decode')
        ref=np.stack(frames);np.save(p/'reference.npy',ref)
        if not (p/'original.mp4').exists(): encode(p,'original',ref,source_fps/2)
        save_meta(p,dict(**video,fps=source_fps/2,count=len(ref),width=320,height=180,synthetic=False,
                        source_fps=source_fps,source_indices=indices,source_timestamps=[i/source_fps for i in indices],
                        reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),reference_note='Decoded MP4 mirror, letterboxed and sampled every second frame; not camera RAW.'))
    stop.clear()
    thread = threading.Thread(target=janitor,daemon=True); thread.start()
    yield
    stop.set(); thread.join(timeout=2)

app = FastAPI(title='LACUNAFRAME',version='0.3.0',lifespan=lifespan)


class Scenario(BaseModel):
    source: str
    seed: int = Field(42,ge=0,le=2**32-1)
    drop: float = Field(.15,ge=0,le=.8)
    gap: int = Field(3,ge=0,le=48)
    start: float = Field(.4,ge=0,le=1)
    occlusion: float = Field(0,ge=0,le=.7)
    mask_shape: Literal['rectangle','brush'] = 'rectangle'
    noise: float = Field(0,ge=0,le=60)
    blur: float = Field(0,ge=0,le=3)


class Run(BaseModel):
    scenario: str
    methods: list[Literal['hold','linear','flow','telea','rife','rife_fast']] = Field(default=['hold','linear','flow'],min_length=1,max_length=6)


@app.get('/api/demos')
def demos() -> list[dict]:
    return [meta(i) for i in NATURAL_IDS if (ROOT/i).exists()]+[meta(f'demo{i}') for i in range(3)]


@app.get('/api/benchmarks')
def benchmarks() -> dict:
    path=Path(__file__).resolve().parents[1]/'examples/natural-v02/summary.json'
    if not path.exists(): return dict(summary=[],protocol='Natural-video benchmark has not been generated yet.')
    report=json.loads(path.read_text())
    return dict(summary=report['summary'],protocol=report['protocol'])


@app.get('/api/capabilities')
def capabilities() -> dict:
    return dict(model=model_info(),natural_sources=MANIFEST['videos'])


@app.post('/api/upload')
async def upload(files: list[UploadFile] = File(...)) -> dict:
    if not 1<=len(files)<=96:
        raise HTTPException(400,'Choose one MP4/MOV or 2–96 PNG/JPEG images (lexical filename order, 12 fps).')
    ident,p = create(); total=0
    try:
        images = len(files)>1 or files[0].content_type in ('image/png','image/jpeg')
        decoded=[]; image_shape=None
        for f in sorted(files,key=lambda f:f.filename or ''):
            allowed = ('image/png','image/jpeg') if images else ('video/mp4','video/quicktime')
            if f.content_type not in allowed:
                raise ValueError('Unsupported MIME type. Use MP4/MOV or PNG/JPEG sequence.')
            data=bytearray()
            while chunk := await f.read(1024*1024):
                total+=len(chunk)
                if total>30*1024*1024:
                    raise ValueError('Upload exceeds 30 MiB total.')
                data.extend(chunk)
            if images:
                # Inspect image header before allocating decoded pixels.
                from PIL import Image
                with Image.open(io.BytesIO(data)) as im:
                    if im.width*im.height>1920*1080:
                        raise ValueError('Image resolution exceeds 1920×1080.')
                a=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
                if a is None: raise ValueError('Image decode failed.')
                if image_shape is not None and image_shape != a.shape: raise ValueError('Sequence images must have matching dimensions.')
                image_shape=a.shape
                decoded.append(normalize(a))
            else:
                # FFmpeg selects the demuxer from the suffix; derive it from the validated MIME type, never the client filename.
                raw=p/('input.mov' if f.content_type=='video/quicktime' else 'input.mp4'); raw.write_bytes(data)
                try:
                    try:
                        info=decode.probe(raw)
                    except Exception as exc:
                        raise ValueError('Video must be decodable, ≤8 seconds and ≤1920×1080.') from exc
                    fps,count=info.fps,info.frame_count
                    if not np.isfinite(fps) or fps<=0 or count<2 or count/fps>8.1 or info.width*info.height>1920*1080:
                        raise ValueError('Video must be decodable, ≤8 seconds and ≤1920×1080.')
                    target=min(12.,fps); next_time=0.; index=0
                    try:
                        for a in decode.frames(raw):
                            if index>2000: raise ValueError('Input frame limit exceeded.')
                            if index/fps+1e-6>=next_time:
                                # Uploaded images arrive as BGR from cv2.imdecode; keep one channel order.
                                decoded.append(normalize(cv2.cvtColor(a,cv2.COLOR_RGB2BGR))); next_time+=1/target
                            index+=1
                            if len(decoded)>96: raise ValueError('More than 96 sampled frames.')
                    except ValueError:
                        raise
                    except Exception as exc:
                        raise ValueError('Video is truncated or partially undecodable.') from exc
                    if np.isfinite(count) and index < count-1: raise ValueError('Video is truncated or partially undecodable.')
                    fps=target
                finally:
                    raw.unlink(missing_ok=True)
        if len(decoded)<2: raise ValueError('At least two decoded frames required.')
        if images:
            if len({a.shape for a in decoded})!=1: raise ValueError('Sequence images must have matching dimensions.')
            fps=12.
        # Letterbox to preserve aspect ratio; video path currently standardized during decode.
        frames=[]
        for a in decoded:
            h,w=a.shape[:2]; scale=min(320/w,180/h)
            resized=cv2.resize(a,(max(1,round(w*scale)),max(1,round(h*scale))))
            canvas=np.zeros((180,320,3),np.uint8); hh,ww=resized.shape[:2]
            canvas[(180-hh)//2:(180-hh)//2+hh,(320-ww)//2:(320-ww)//2+ww]=resized
            frames.append(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
        ref=np.stack(frames); np.save(p/'reference.npy',ref)
        info=dict(id=ident,name='Local upload',fps=fps,count=len(ref),width=320,height=180,synthetic=False,sha256=hashlib.sha256(ref.tobytes()).hexdigest())
        save_meta(p,info)
        return info
    except Exception as exc:
        shutil.rmtree(p,ignore_errors=True)
        if isinstance(exc,HTTPException): raise
        raise HTTPException(400,str(exc)) from exc
    finally:
        for f in files: await f.close()


@app.post('/api/scenarios')
def scenario(config: Scenario) -> dict:
    source=locate(config.source)
    ref=np.load(source/'reference.npy')
    ident,p=create()
    damaged,missing,mask=corrupt(ref,config.model_dump())
    for name,array in [('reference',ref),('corrupted',damaged),('missing',missing),('mask',mask)]: np.save(p/f'{name}.npy',array)
    info={**meta(config.source),'id':ident,'config':config.model_dump(),'missing':missing.tolist(),'sha256':hashlib.sha256(ref.tobytes()).hexdigest()}
    save_meta(p,info)
    return info


def process(ident: str, source: Path, methods: list[str]) -> None:
    p=locate(ident)
    try:
        ref=np.load(source/'reference.npy'); damaged=np.load(source/'corrupted.npy'); missing=np.load(source/'missing.npy'); mask=np.load(source/'mask.npy')
        info=json.loads((p/'meta.json').read_text()); fps=info['fps']
        affected=mask.copy(); affected[missing]=True
        if info['config']['noise'] or info['config']['blur']: affected[:]=True
        report=dict(version='0.3.0',environment=dict(python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,skimage=importlib.metadata.version('scikit-image')),source=info,metric_notes='PSNR from pooled MSE; Infinity string means exact match. SSIM is mean local RGB SSIM; masked SSIM includes neighborhoods outside mask. Temporal error = mean absolute error in consecutive-frame differences / 255. Missing corrupted frames use zero placeholders for metric penalty only.',corrupted=evaluate(ref,damaged,affected,fps),methods={})
        display=damaged.copy()
        for i in np.flatnonzero(missing):
            display[i][:]=(30,28,35)
            cv2.putText(display[i],'MISSING SAMPLE',(43,94),cv2.FONT_HERSHEY_SIMPLEX,.6,(250,140,120),1,cv2.LINE_AA)
        report['corrupted']['media']=encode(p,'corrupted',display,fps)
        for j,method in enumerate(methods):
            jobs[ident]['stage']='Loading '+method
            loading=time.perf_counter()
            if method.startswith('rife'):session()
            loading=time.perf_counter()-loading
            memory=[psutil.Process().memory_info().rss]
            sampling=threading.Event()
            def sample_memory():
                while not sampling.wait(.02): memory.append(psutil.Process().memory_info().rss)
            sampler=threading.Thread(target=sample_memory,daemon=True);sampler.start()
            begin=time.perf_counter()
            def update(fraction): jobs[ident]['progress']=(j+fraction)/len(methods)*.9
            jobs[ident]['stage']='Restoring with '+method
            try:
                out,warnings,uncertainty=restore(damaged,missing,mask,method,update)
                runtime=time.perf_counter()-begin
            finally:
                sampling.set();sampler.join()
            jobs[ident]['stage']='Evaluating and encoding '+method
            np.save(p/f'{method}.npy',out)
            metrics=evaluate(ref,out,affected,fps)
            media=encode(p,method,out,fps)
            report['methods'][method]=dict(**metrics,runtime_seconds=runtime,model_initialization_seconds=loading,process_peak_rss_mib=max(memory)/1024**2,model=model_info() if method.startswith('rife') else None,inference_scale=.5 if method=='rife_fast' else 1.,warnings=warnings,reliability_available=method=='flow',uncertainty=uncertainty if method=='flow' else None,media=media)
        (p/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        with (p/'metrics.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['method','frame','time','mse','psnr','ssim','affected_psnr','affected_ssim']); writer.writeheader()
            for method,result in {'corrupted':report['corrupted'],**report['methods']}.items():
                writer.writerows(dict(method=method,**row) for row in result['frames'])
        jobs[ident].update(status='complete',progress=1)
        log.info(json.dumps(dict(event='run_complete',id=ident,methods=methods)))
    except Exception:
        log.exception(json.dumps(dict(event='run_failed',id=ident)))
        jobs[ident].update(status='failed',error='Processing failed; inspect backend logs.')
    finally:
        gate.release()


@app.post('/api/run')
def run(request: Run) -> dict:
    source=locate(request.scenario)
    if not (source/'corrupted.npy').exists(): raise HTTPException(400,'Create a scenario first')
    if any(m.startswith('rife') for m in request.methods) and not model_info()['available']:
        raise HTTPException(503,'RIFE weights unavailable; run python scripts/fetch_assets.py')
    if not gate.acquire(blocking=False): raise HTTPException(429,'CPU worker busy. Wait for the current run.')
    try:
        ident,p=create()
        info={**meta(request.scenario),'id':ident,'scenario':request.scenario}
        save_meta(p,info)
        # Self-contained run: survives source expiration.
        for name in ('reference','corrupted','missing','mask'): shutil.copyfile(source/f'{name}.npy',p/f'{name}.npy')
        jobs[ident]=dict(id=ident,status='running',progress=0)
        executor.submit(process,ident,p,list(dict.fromkeys(request.methods)))
        return jobs[ident]
    except Exception:
        gate.release(); raise


@app.get('/api/results/{ident}')
def results(ident: str) -> dict:
    p=locate(ident)
    if (p/'report.json').exists(): return dict(id=ident,status='complete',progress=1,report=json.loads((p/'report.json').read_text()))
    return jobs.get(ident,dict(id=ident,status='failed',error='Run interrupted by server restart'))


@app.get('/api/frames/{ident}/{kind}/{index}')
def frame(ident: str,kind: str,index: int,method: str='flow') -> Response:
    p=locate(ident)
    if kind not in ('original','corrupted','hold','linear','flow','telea','rife','rife_fast','mask','error'): raise HTTPException(404,'Unknown view')
    name='reference' if kind=='original' else kind
    if kind=='error':
        if method not in ('hold','linear','flow','telea','rife','rife_fast'): raise HTTPException(404,'Unknown method')
        name=method
    path=p/f'{name}.npy'
    if not path.exists(): raise HTTPException(404,'View not available yet')
    arr=np.load(path,mmap_mode='r')
    if not 0<=index<len(arr): raise HTTPException(404,'Frame out of bounds')
    a=arr[index].copy()
    if kind=='error':
        ref=np.load(p/'reference.npy',mmap_mode='r')[index]
        a=cv2.applyColorMap(np.abs(a.astype(float)-ref.astype(float)).mean(-1).astype('uint8'),cv2.COLORMAP_INFERNO)
    elif kind=='mask':
        if np.load(p/'missing.npy')[index]: a[:]=True
        a=np.repeat((a.astype('uint8')*255)[...,None],3,-1)
    else:
        a=cv2.cvtColor(a,cv2.COLOR_RGB2BGR)
        if kind=='corrupted' and np.load(p/'missing.npy')[index]:
            a[:]=(35,28,30)
            cv2.putText(a,'MISSING SAMPLE',(43,94),cv2.FONT_HERSHEY_SIMPLEX,.6,(120,140,250),1,cv2.LINE_AA)
    ok,data=cv2.imencode('.png',a)
    return Response(data.tobytes(),media_type='image/png',headers={'Cache-Control':'private, max-age=300'})


@app.get('/api/download/{ident}/{filename}')
def download(ident: str,filename: str) -> FileResponse:
    allowed={'report.json','metrics.csv'}|{f'{m}.{ext}' for m in ('hold','linear','flow','telea','rife','rife_fast','original','corrupted') for ext in ('mp4','gif')}
    if filename not in allowed: raise HTTPException(404,'Unknown artifact')
    p=locate(ident)/filename
    if not p.exists(): raise HTTPException(404,'Artifact unavailable')
    return FileResponse(p,filename=filename)


# Optional single-container mode (e.g. a Hugging Face Space): serve the built
# frontend from the same origin. Mounted last so every /api route takes precedence.
if os.environ.get('LACUNA_STATIC'):
    from fastapi.staticfiles import StaticFiles
    app.mount('/', StaticFiles(directory=os.environ['LACUNA_STATIC'], html=True), name='frontend')
