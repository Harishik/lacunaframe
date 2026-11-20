"""Pixel-domain corruption, reconstruction baselines and evaluation.

Conventions
-----------
* Frames are RGB uint8 arrays shaped (N, H, W, 3).
* ``missing`` is a boolean (N,) vector of absent temporal samples. Missing
  frames are stored as zeros, but that placeholder is never read as data.
* ``mask`` is a boolean (N, H, W) array of spatially hidden pixels.
* Solvers receive observations only. The clean reference reaches
  ``evaluate`` and nothing else.
"""
from __future__ import annotations
import cv2
import numpy as np
from skimage.metrics import structural_similarity

EXTERNAL_METHODS = ('amt_s', 'ifrnet_s', 'ifrnet', 'rife_paper', 'amt_s_gopro', 'ifrnet_s_gopro', 'ifrnet_gopro')  # published models via backend/external.py (ONNX); plain names = Vimeo90K (midpoint-trained) weights
METHODS = ('hold', 'linear', 'flow', 'telea', 'rife', 'rife_fast', 'flow_quadratic', *EXTERNAL_METHODS)
# Research-only methods: available to benchmarks, not yet exposed through the API/UI.
RESEARCH_METHODS = ('flow_quadratic', *EXTERNAL_METHODS)
LONG_GAP = 6  # anchor separation above which a gap is reported as underconstrained


# ---------------------------------------------------------------------------
# Synthetic controls
# ---------------------------------------------------------------------------

def demo(kind: int, n: int = 48, h: int = 180, w: int = 320) -> np.ndarray:
    """Deterministic synthetic scene: textured background, a moving disc, and
    (kind > 0) a counter-moving striped box; kind 2 also pans the background."""
    rng = np.random.default_rng(2026)
    y, x = np.mgrid[:h, :w]
    texture = rng.integers(0, 30, (h, w, 1))
    gradient = np.stack([22 + x * .08, 30 + y * .12, 45 + x * .09], -1)
    base = np.clip(gradient + texture, 0, 255).astype('uint8')
    frames = []
    for i in range(n):
        frame = np.roll(base, i * 2 if kind == 2 else 0, axis=1).copy()
        for xx in range(0, w, 32):
            cv2.line(frame, (xx, 0), (xx, h), (43, 54, 68), 1)
        t = i / (n - 1)
        cx = int(38 + t * (w - 76))
        cy = int(h * .48 + 20 * np.sin(t * 6))
        cv2.circle(frame, (cx, cy), 24, (244, 118, 98), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx - 7, cy - 8), 5, (255, 194, 166), -1)
        if kind > 0:
            bx = int(w - 48 - t * (w - 96))
            cv2.rectangle(frame, (bx - 22, 86), (bx + 22, 130), (75, 155, 242), -1)
            for z in range(-15, 20, 7):
                cv2.line(frame, (bx + z, 89), (bx + z, 126), (115, 191, 255), 2)
        frames.append(frame)
    return np.stack(frames)


# ---------------------------------------------------------------------------
# Corruption operator  y = M B x + n
# ---------------------------------------------------------------------------

def _spatial_mask(h: int, w: int, size: float, shape: str | None) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    if shape == 'brush':
        cv2.line(m, (int(w * .3), int(h * .35)), (int(w * .7), int(h * .65)), 1, max(1, int(h * size)))
    else:
        dx, dy = max(1, int(w * size / 2)), max(1, int(h * size / 2))
        m[h // 2 - dy:h // 2 + dy, w // 2 - dx:w // 2 + dx] = 1
    return m.astype(bool)


def corrupt(ref: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply seeded temporal sampling, spatial masking, noise and blur.

    Returns (damaged frames, missing flags, spatial mask). The random-number
    call order is part of the reproducibility contract: drops, then noise.
    """
    rng = np.random.default_rng(config['seed'])
    out = ref.copy()
    n, h, w, _ = ref.shape

    missing = rng.random(n) < config['drop']
    start = min(n - 1, int(config['start'] * n))
    missing[start:min(n, start + config['gap'])] = True
    if missing.all():  # keep one anchor even at extreme settings
        missing[0] = False

    mask = np.zeros((n, h, w), bool)
    if config['occlusion']:
        mask[:] = _spatial_mask(h, w, config['occlusion'], config.get('mask_shape'))
        out[mask] = 0
    if config['noise']:
        noisy = out.astype(float) + rng.normal(0, config['noise'], out.shape)
        out = np.clip(noisy, 0, 255).astype('uint8')
    if config['blur']:
        out = np.stack([cv2.GaussianBlur(a, (5, 5), config['blur']) for a in out])
    out[missing] = 0  # storage placeholder only; never used as an observed anchor
    return out, missing, mask


# ---------------------------------------------------------------------------
# Motion-compensated interpolation (two anchors)
# ---------------------------------------------------------------------------

FARNEBACK = (.5, 3, 21, 4, 7, 1.5, 0)  # pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
CONSISTENCY_PX = 1.5
FIXED_POINT_STEPS = 5


def flow_between(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense Farnebäck flow F_ab (a -> b) and F_ba (b -> a) on grayscale."""
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return (cv2.calcOpticalFlowFarneback(ga, gb, None, *FARNEBACK),
            cv2.calcOpticalFlowFarneback(gb, ga, None, *FARNEBACK))


def _pixel_grid(h: int, w: int) -> np.ndarray:
    return np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype('float32')


def _inside(points: np.ndarray, h: int, w: int) -> np.ndarray:
    return (points[..., 0] >= 0) & (points[..., 0] <= w - 1) & (points[..., 1] >= 0) & (points[..., 1] <= h - 1)


def _warp_to_time(src: np.ndarray, flow: np.ndarray, reverse: np.ndarray, fraction: float,
                  grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample `src` at time `fraction` along `flow`.

    For each destination pixel q, solve p + fraction·F(p) = q by fixed-point
    iteration (remap is a destination -> source lookup). A path is valid when
    it stays in bounds and the forward/backward flows agree to CONSISTENCY_PX.
    """
    h, w = src.shape[:2]
    p = grid.copy()
    for _ in range(FIXED_POINT_STEPS):
        p = grid - fraction * cv2.remap(flow, p, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    forward = cv2.remap(flow, p, None, cv2.INTER_LINEAR)
    end = p + forward
    backward = cv2.remap(reverse, end, None, cv2.INTER_LINEAR)
    consistent = np.linalg.norm(forward + backward, axis=-1) < CONSISTENCY_PX
    valid = consistent & _inside(p, h, w) & _inside(end, h, w)
    return cv2.remap(src, p, None, cv2.INTER_LINEAR).astype(float), valid


def motion(a: np.ndarray, b: np.ndarray, t: float) -> tuple[np.ndarray, float]:
    """Bidirectional flow interpolation between anchors a (t=0) and b (t=1).

    Valid warps are blended with temporal weights. Where neither path is
    reliable the pixel falls back to linear blending. The returned fraction of
    pixels lacking two reliable paths is a consistency heuristic, not a posterior.
    """
    f_ab, f_ba = flow_between(a, b)
    grid = _pixel_grid(*a.shape[:2])
    warped_a, valid_a = _warp_to_time(a, f_ab, f_ba, t, grid)
    warped_b, valid_b = _warp_to_time(b, f_ba, f_ab, 1 - t, grid)
    weight = (1 - t) * valid_a + t * valid_b
    blended = ((1 - t) * warped_a * valid_a[..., None] + t * warped_b * valid_b[..., None])
    merged = np.divide(blended, np.maximum(weight[..., None], 1e-9))
    fallback = (1 - t) * a.astype(float) + t * b.astype(float)
    merged[weight == 0] = fallback[weight == 0]
    return np.clip(merged, 0, 255).round().astype('uint8'), float(np.mean(~(valid_a & valid_b)))


# ---------------------------------------------------------------------------
# Multi-anchor (quadratic-motion) interpolation
# ---------------------------------------------------------------------------

def quadratic_displacement(f_next: np.ndarray, f_prev: np.ndarray, s_next: float, s_prev: float,
                           tau: float) -> np.ndarray:
    """Displacement at time tau of a constant-acceleration trajectory.

    The anchor sits at time 0, the next anchor at +s_next (displacement
    f_next), and the previous one at -s_prev (displacement f_prev). Solving
    d(s) = v s + a s^2 / 2 through both gives

        a = 2 (f_next s_prev + f_prev s_next) / (s_prev s_next (s_prev + s_next))
        v = (f_next - a s_next^2 / 2) / s_next

    For uniform motion (f_prev = -f_next s_prev / s_next), a = 0 and d(tau) is linear.
    """
    accel = 2 * (f_next * s_prev + f_prev * s_next) / (s_prev * s_next * (s_prev + s_next))
    velocity = (f_next - accel * s_next ** 2 / 2) / s_next
    return velocity * tau + accel * tau ** 2 / 2


def _warp_by_displacement(src: np.ndarray, displacement: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backward-warp `src` so pixel q takes the source p with p + D(p) = q."""
    h, w = src.shape[:2]
    p = grid.copy()
    for _ in range(FIXED_POINT_STEPS):
        p = grid - cv2.remap(displacement, p, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.remap(src, p, None, cv2.INTER_LINEAR).astype(float), _inside(p, h, w), p


class QuadraticFlow:
    """Quadratic-motion interpolation using up to two observed anchors per side.

    Flows are computed once per bracket and reused for every missing frame in
    that gap. When an outer anchor is missing (sequence boundary), that side
    falls back to the linear trajectory of the two-anchor method.
    """

    def __init__(self, frames: np.ndarray, observed: np.ndarray):
        self.frames, self.observed = frames, observed
        self._cache: dict[tuple[int, int], dict] = {}

    def _bracket(self, left: int, right: int) -> dict:
        if (left, right) not in self._cache:
            before = self.observed[self.observed < left]
            after = self.observed[self.observed > right]
            a, b = self.frames[left], self.frames[right]
            f_ab, f_ba = flow_between(a, b)
            entry = dict(f_ab=f_ab, f_ba=f_ba, outer_left=None, outer_right=None)
            if len(before):
                entry['outer_left'] = (int(before[-1]), flow_between(a, self.frames[before[-1]])[0])
            if len(after):
                entry['outer_right'] = (int(after[0]), flow_between(b, self.frames[after[0]])[0])
            self._cache[(left, right)] = entry
        return self._cache[(left, right)]

    def __call__(self, left: int, right: int, i: int) -> tuple[np.ndarray, float]:
        e = self._bracket(left, right)
        gap = right - left
        t = (i - left) / gap
        a, b = self.frames[left], self.frames[right]
        h, w = a.shape[:2]
        grid = _pixel_grid(h, w)
        # Displacement from each anchor to the target time (in frames).
        if e['outer_left'] is not None:
            j, f_prev = e['outer_left']
            d_a = quadratic_displacement(e['f_ab'], f_prev, gap, left - j, i - left)
        else:
            d_a = t * e['f_ab']
        if e['outer_right'] is not None:
            j, f_next = e['outer_right']
            # Anchor b at time 0: the outer anchor j is ahead (+(j-right)), anchor a behind (-gap).
            d_b = quadratic_displacement(f_next, e['f_ba'], j - right, gap, -(right - i))
        else:
            d_b = (1 - t) * e['f_ba']
        warped_a, inside_a, p_a = _warp_by_displacement(a, d_a.astype('float32'), grid)
        warped_b, inside_b, p_b = _warp_by_displacement(b, d_b.astype('float32'), grid)
        # Reliability: forward/backward consistency of the two-anchor flows at the sampled sources.
        fwd = cv2.remap(e['f_ab'], p_a, None, cv2.INTER_LINEAR)
        back = cv2.remap(e['f_ba'], p_a + fwd, None, cv2.INTER_LINEAR)
        valid_a = inside_a & (np.linalg.norm(fwd + back, axis=-1) < CONSISTENCY_PX)
        fwd_b = cv2.remap(e['f_ba'], p_b, None, cv2.INTER_LINEAR)
        back_b = cv2.remap(e['f_ab'], p_b + fwd_b, None, cv2.INTER_LINEAR)
        valid_b = inside_b & (np.linalg.norm(fwd_b + back_b, axis=-1) < CONSISTENCY_PX)
        weight = (1 - t) * valid_a + t * valid_b
        blended = (1 - t) * warped_a * valid_a[..., None] + t * warped_b * valid_b[..., None]
        merged = np.divide(blended, np.maximum(weight[..., None], 1e-9))
        fallback = (1 - t) * a.astype(float) + t * b.astype(float)
        merged[weight == 0] = fallback[weight == 0]
        return np.clip(merged, 0, 255).round().astype('uint8'), float(np.mean(~(valid_a & valid_b)))


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def _interpolate(method: str, frames: np.ndarray, left: int, right: int, t: float,
                 quadratic: QuadraticFlow | None = None, index: int | None = None) -> tuple[np.ndarray, float]:
    """Estimate one missing frame between observed anchors `left` and `right`."""
    if method == 'flow_quadratic':
        return quadratic(left, right, index)
    a, b = frames[left], frames[right]
    if method in ('hold', 'telea'):
        return frames[left if t <= .5 else right], 0.0
    if method == 'linear':
        return ((1 - t) * a.astype(float) + t * b).round().astype('uint8'), 0.0
    if method in EXTERNAL_METHODS:
        from .external import interpolate as external
        return external(method, a, b, t), 0.0
    if method.startswith('rife'):
        from .learned import interpolate
        return interpolate(a, b, t, fast=method == 'rife_fast'), 0.0
    return motion(a, b, t)


def restore(damaged: np.ndarray, missing: np.ndarray, mask: np.ndarray, method: str,
            progress=None) -> tuple[np.ndarray, list[str], list[float]]:
    """Reconstruct missing frames (and, for Telea, masked pixels) from observations.

    Returns (frames, sorted unique warnings, per-frame unreliability in [0, 1]).
    """
    if method not in METHODS:
        raise ValueError('Unknown method')
    observed = np.flatnonzero(~missing)
    if not len(observed):
        raise ValueError('No observed frames: recovery is impossible')
    out = damaged.copy()
    warnings: list[str] = []
    unreliability = [0.0] * len(out)

    if method == 'telea':
        for i in observed:
            if mask[i].any():
                out[i] = cv2.inpaint(out[i], mask[i].astype('uint8') * 255, 3, cv2.INPAINT_TELEA)
        warnings.append('Telea fills spatial masks only; missing frames use nearest observed frame.')
    elif mask.any():
        warnings.append('Temporal methods leave within-frame masks unrepaired. Choose Telea for spatial inpainting.')
    if method.startswith('rife') and method not in EXTERNAL_METHODS:
        warnings.append('Pretrained RIFE estimates missing frames; it cannot verify hidden content. No calibrated confidence is available.')
        if method == 'rife_fast':
            warnings.append('RIFE Fast runs at half spatial resolution then upsamples; fine detail may be lost.')

    quadratic = QuadraticFlow(out, observed) if method == 'flow_quadratic' else None
    for i in range(len(out)):
        if missing[i]:
            before, after = observed[observed < i], observed[observed > i]
            if not len(before) or not len(after):
                out[i] = out[observed[np.argmin(abs(observed - i))]]
                unreliability[i] = 1.0
                warnings.append('Boundary missing frames have no bracket; nearest observed frame used.')
            else:
                left, right = int(before[-1]), int(after[0])
                if right - left > LONG_GAP:
                    warnings.append('Long gap (>5 frames): motion and visibility are underconstrained.')
                out[i], unreliability[i] = _interpolate(method, out, left, right, (i - left) / (right - left), quadratic, i)
        if progress:
            progress((i + 1) / len(out))
    warnings.append('No denoising or deblurring is implemented; those stimuli may remain in the result.')
    return out, sorted(set(warnings)), unreliability


# ---------------------------------------------------------------------------
# Evaluation (reference-aware; never called by solvers)
# ---------------------------------------------------------------------------

def psnr(mse: float) -> float | str:
    """PSNR in dB for 8-bit data; exact match is the JSON-safe string 'Infinity'."""
    return 'Infinity' if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def _validate(ref: np.ndarray, result: np.ndarray, affected: np.ndarray, fps: float) -> None:
    if ref.ndim != 4 or ref.shape[-1] != 3 or min(ref.shape) < 1 or ref.shape != result.shape:
        raise ValueError('Aligned nonempty NHWC RGB frames required')
    if not np.isfinite(ref).all() or not np.isfinite(result).all():
        raise ValueError('Aligned finite frames required')
    if affected.shape != ref.shape[:3] or affected.dtype != np.bool_:
        raise ValueError('Boolean affected mask aligned with frames required')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Positive finite FPS required')


def _ssim_window(h: int, w: int) -> int:
    """Largest odd window <= 7 that fits the frame (< 3 means SSIM is undefined)."""
    win = min(7, h, w)
    return win - (win + 1) % 2


def evaluate(ref: np.ndarray, result: np.ndarray, affected: np.ndarray, fps: float) -> dict:
    """Whole-frame and affected-support PSNR/SSIM, per-frame rows and temporal error."""
    _validate(ref, result, affected, fps)
    error = (ref.astype(float) - result.astype(float)) ** 2
    win = _ssim_window(ref.shape[1], ref.shape[2])
    rows, ssim_maps = [], []
    for i, (a, b) in enumerate(zip(ref, result)):
        if win >= 3:
            score, full = structural_similarity(a, b, channel_axis=-1, data_range=255, win_size=win, full=True)
            full = full.mean(axis=-1)
        else:
            score, full = None, None
        ssim_maps.append(full)
        m = affected[i]
        rows.append(dict(
            frame=i, time=i / fps, mse=float(error[i].mean()), psnr=psnr(float(error[i].mean())),
            ssim=None if score is None else float(score),
            affected_psnr=psnr(float(error[i][m].mean())) if m.any() else None,
            affected_ssim=float(full[m].mean()) if m.any() and full is not None else None))

    changed = affected.any(axis=(1, 2))
    temporal = np.diff(result.astype(float), axis=0) - np.diff(ref.astype(float), axis=0)
    ssim_values = [r['ssim'] for r in rows if r['ssim'] is not None]
    if win >= 3 and affected.any():
        masked_ssim = np.concatenate([s[a] for s, a in zip(ssim_maps, affected) if s is not None and a.any()])
    else:
        masked_ssim = np.array([])
    return dict(
        psnr=psnr(float(error.mean())),
        ssim=float(np.mean(ssim_values)) if ssim_values else None,
        affected_psnr=psnr(float(error[affected].mean())) if affected.any() else None,
        affected_ssim=float(masked_ssim.mean()) if masked_ssim.size else None,
        affected_frames_psnr=psnr(float(error[changed].mean())) if changed.any() else None,
        affected_frames_ssim=float(np.mean([rows[i]['ssim'] for i in np.flatnonzero(changed)])) if changed.any() and win >= 3 else None,
        temporal_error=float(np.abs(temporal).mean() / 255) if len(ref) > 1 else 0.,
        frames=rows)
