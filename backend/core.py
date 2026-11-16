"""Pixel-domain baselines. RGB uint8 arrays, explicit missing-sample flags."""
from __future__ import annotations
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity

METHODS = ('hold', 'linear', 'flow', 'telea', 'rife', 'rife_fast')


def demo(kind: int, n: int = 48, h: int = 180, w: int = 320) -> np.ndarray:
    rng = np.random.default_rng(2026)
    y, x = np.mgrid[:h, :w]
    texture = rng.integers(0, 30, (h, w, 1))
    base = np.clip(np.stack([22+x*.08, 30+y*.12, 45+x*.09], -1)+texture, 0, 255).astype('uint8')
    frames = []
    for i in range(n):
        a = np.roll(base, i*2 if kind == 2 else 0, axis=1).copy()
        for xx in range(0, w, 32):
            cv2.line(a, (xx, 0), (xx, h), (43, 54, 68), 1)
        t = i/(n-1)
        cx = int(38+t*(w-76))
        cv2.circle(a, (cx, int(h*.48+20*np.sin(t*6))), 24, (244, 118, 98), -1, cv2.LINE_AA)
        cv2.circle(a, (cx-7, int(h*.48+20*np.sin(t*6))-8), 5, (255, 194, 166), -1)
        if kind > 0:
            bx = int(w-48-t*(w-96))
            cv2.rectangle(a, (bx-22, 86), (bx+22, 130), (75, 155, 242), -1)
            for z in range(-15, 20, 7):
                cv2.line(a, (bx+z, 89), (bx+z, 126), (115, 191, 255), 2)
        frames.append(a)
    return np.stack(frames)


def corrupt(ref: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config['seed'])
    out = ref.copy()
    n, h, w, _ = ref.shape
    missing = rng.random(n) < config['drop']
    start = min(n-1, int(config['start']*n))
    missing[start:min(n, start+config['gap'])] = True
    # Keep one anchor even at extreme settings.
    if missing.all():
        missing[0] = False
    mask = np.zeros((n, h, w), bool)
    size = config['occlusion']
    if size:
        m = np.zeros((h,w), np.uint8)
        if config.get('mask_shape') == 'brush':
            cv2.line(m, (int(w*.3), int(h*.35)), (int(w*.7), int(h*.65)), 1, max(1,int(h*size)))
        else:
            dx, dy = max(1,int(w*size/2)), max(1,int(h*size/2))
            m[h//2-dy:h//2+dy,w//2-dx:w//2+dx] = 1
        mask[:] = m.astype(bool)
        out[mask] = 0
    if config['noise']:
        out = np.clip(out.astype(float)+rng.normal(0,config['noise'],out.shape),0,255).astype('uint8')
    if config['blur']:
        out = np.stack([cv2.GaussianBlur(a,(5,5),config['blur']) for a in out])
    out[missing] = 0  # Storage placeholder only; never used as an observed anchor.
    return out, missing, mask


def flow_between(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = lambda v: cv2.cvtColor(v, cv2.COLOR_RGB2GRAY)
    params = (.5, 3, 21, 4, 7, 1.5, 0)
    return (cv2.calcOpticalFlowFarneback(gray(a),gray(b),None,*params),
            cv2.calcOpticalFlowFarneback(gray(b),gray(a),None,*params))


def motion(a: np.ndarray, b: np.ndarray, t: float) -> tuple[np.ndarray, float]:
    """F(x): A->B; B(x): B->A. Invert x+tF(x)=q by fixed-point iteration.
    remap is destination->source. Reject inconsistent/out-of-bounds paths;
    use linear blend where neither path is reliable. This is a heuristic, not a posterior.
    """
    f, back = flow_between(a,b)
    h,w = a.shape[:2]
    grid = np.stack(np.meshgrid(np.arange(w),np.arange(h)), -1).astype('float32')
    def warp(src: np.ndarray, flow: np.ndarray, reverse: np.ndarray, fraction: float):
        p = grid.copy()
        for _ in range(5):
            p = grid-fraction*cv2.remap(flow,p,None,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        ff = cv2.remap(flow,p,None,cv2.INTER_LINEAR)
        end = p+ff
        rr = cv2.remap(reverse,end,None,cv2.INTER_LINEAR)
        valid = (np.linalg.norm(ff+rr,axis=-1)<1.5) & (p[...,0]>=0)&(p[...,0]<=w-1)&(p[...,1]>=0)&(p[...,1]<=h-1)&(end[...,0]>=0)&(end[...,0]<=w-1)&(end[...,1]>=0)&(end[...,1]<=h-1)
        return cv2.remap(src,p,None,cv2.INTER_LINEAR).astype(float), valid
    wa,va = warp(a,f,back,t)
    wb,vb = warp(b,back,f,1-t)
    weight = (1-t)*va+t*vb
    fallback = (1-t)*a.astype(float)+t*b.astype(float)
    merged = np.divide((1-t)*wa*va[...,None]+t*wb*vb[...,None],np.maximum(weight[...,None],1e-9))
    merged[weight==0] = fallback[weight==0]
    return np.clip(merged,0,255).round().astype('uint8'), float(np.mean(~(va&vb)))


def restore(damaged: np.ndarray, missing: np.ndarray, mask: np.ndarray, method: str, progress=None) -> tuple[np.ndarray, list[str], list[float]]:
    if method not in METHODS:
        raise ValueError('Unknown method')
    valid = np.flatnonzero(~missing)
    if not len(valid):
        raise ValueError('No observed frames: recovery is impossible')
    out = damaged.copy()
    warnings = []
    uncertainty = [0.0]*len(out)
    if method == 'telea':
        for i in valid:
            if mask[i].any():
                out[i] = cv2.inpaint(out[i],mask[i].astype('uint8')*255,3,cv2.INPAINT_TELEA)
        warnings.append('Telea fills spatial masks only; missing frames use nearest observed frame.')
    elif mask.any():
        warnings.append('Temporal methods leave within-frame masks unrepaired. Choose Telea for spatial inpainting.')
    if method.startswith('rife'):
        from .learned import interpolate
        warnings.append('Pretrained RIFE estimates missing frames; it cannot verify hidden content. No calibrated confidence is available.')
        if method == 'rife_fast':
            warnings.append('RIFE Fast runs at half spatial resolution then upsamples; fine detail may be lost.')
    for i in range(len(out)):
        if missing[i]:
            left = valid[valid<i]; right = valid[valid>i]
            if not len(left) or not len(right):
                out[i] = out[valid[np.argmin(abs(valid-i))]]
                uncertainty[i] = 1.0
                warnings.append('Boundary missing frames have no bracket; nearest observed frame used.')
            else:
                l,r = int(left[-1]),int(right[0]); t = (i-l)/(r-l)
                if r-l>6:
                    warnings.append('Long gap (>5 frames): motion and visibility are underconstrained.')
                if method in ('hold','telea'):
                    out[i] = out[l if t<=.5 else r]
                elif method == 'linear':
                    out[i] = ((1-t)*out[l].astype(float)+t*out[r]).round().astype('uint8')
                elif method.startswith('rife'):
                    out[i] = interpolate(out[l],out[r],t,fast=method=='rife_fast')
                else:
                    out[i], uncertainty[i] = motion(out[l],out[r],t)
        if progress:
            progress((i+1)/len(out))
    warnings.append('No denoising or deblurring is implemented; those stimuli may remain in the result.')
    return out, sorted(set(warnings)), uncertainty


def psnr(mse: float) -> float | str:
    return 'Infinity' if mse == 0 else float(10*np.log10(255**2/mse))


def evaluate(ref: np.ndarray, result: np.ndarray, affected: np.ndarray, fps: float) -> dict:
    if ref.ndim != 4 or ref.shape[-1] != 3 or min(ref.shape) < 1 or ref.shape != result.shape:
        raise ValueError('Aligned nonempty NHWC RGB frames required')
    if not np.isfinite(ref).all() or not np.isfinite(result).all():
        raise ValueError('Aligned finite frames required')
    if affected.shape != ref.shape[:3] or affected.dtype != np.bool_:
        raise ValueError('Boolean affected mask aligned with frames required')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Positive finite FPS required')
    error = (ref.astype(float)-result.astype(float))**2
    rows = []
    maps = []
    win = min(7,ref.shape[1],ref.shape[2]); win -= (win+1)%2
    for i,(a,b) in enumerate(zip(ref,result)):
        if win >= 3:
            s, sm = structural_similarity(a,b,channel_axis=-1,data_range=255,win_size=win,full=True)
            sm = sm.mean(axis=-1)
        else:
            s,sm = None,None
        maps.append(sm)
        m = affected[i]
        rows.append(dict(frame=i,time=i/fps,mse=float(error[i].mean()),psnr=psnr(float(error[i].mean())),ssim=None if s is None else float(s),affected_psnr=psnr(float(error[i][m].mean())) if m.any() else None,affected_ssim=float(sm[m].mean()) if m.any() and sm is not None else None))
    changed = affected.any(axis=(1,2))
    temporal = np.diff(result.astype(float),axis=0)-np.diff(ref.astype(float),axis=0)
    ssim_values = [r['ssim'] for r in rows if r['ssim'] is not None]
    masked_ssim = np.concatenate([m[a] for m,a in zip(maps,affected) if m is not None and a.any()]) if win>=3 and affected.any() else np.array([])
    return dict(psnr=psnr(float(error.mean())),ssim=float(np.mean(ssim_values)) if ssim_values else None,
                affected_psnr=psnr(float(error[affected].mean())) if affected.any() else None,
                affected_ssim=float(masked_ssim.mean()) if masked_ssim.size else None,
                affected_frames_psnr=psnr(float(error[changed].mean())) if changed.any() else None,
                affected_frames_ssim=float(np.mean([rows[i]['ssim'] for i in np.flatnonzero(changed)])) if changed.any() and win>=3 else None,
                temporal_error=float(np.abs(temporal).mean()/255) if len(ref)>1 else 0.,frames=rows)
