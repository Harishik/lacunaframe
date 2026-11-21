"""Differentiable, parameter-free pyramidal Lucas-Kanade flow (unrolled Gauss-Newton).

Built from conv / pooling / grid_sample only, so it trains end to end and exports to ONNX.
Used as LacunaNet's structural motion prior: correct (classical-quality) motion from step 0,
which the learned matcher then refines. See DESIGN_AUDIT.md section 6.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _warp(x, flow):
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, dtype=x.dtype, device=x.device), torch.arange(w, dtype=x.dtype, device=x.device), indexing='ij')
    px = (gx[None] + flow[:, 0]) * (2.0 / max(w - 1, 1)) - 1.0
    py = (gy[None] + flow[:, 1]) * (2.0 / max(h - 1, 1)) - 1.0
    return F.grid_sample(x, torch.stack([px, py], -1), mode='bilinear', padding_mode='border', align_corners=True)


def _gauss(sigma: float, dtype, device):
    r = max(1, int(math.ceil(2.5 * sigma)))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=dtype, device=device) ** 2 / (2 * sigma ** 2))
    return k / k.sum(), r


def _blur(x, sigma):
    k, r = _gauss(sigma, x.dtype, x.device)
    c = x.shape[1]
    x = F.pad(x, [r, r, 0, 0], mode='replicate')
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, [0, 0, r, r], mode='replicate')
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def _grad(x):
    xp = F.pad(x, [1, 1, 1, 1], mode='replicate')
    return (xp[:, :, 1:-1, 2:] - xp[:, :, 1:-1, :-2]) / 2, (xp[:, :, 2:, 1:-1] - xp[:, :, :-2, 1:-1]) / 2


def luma(img):
    return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]


def lk_flow(a, b, levels: int = 5, iters: int = 3, sigma: float = 3.0, lam: float = 1e-4, out_scale: int = 1):
    """Displacement a -> b (a(p) ~ b(p + f)) in pixels of the 1/out_scale grid. a, b: N x 3 x H x W in [0, 1]."""
    ya, yb = luma(a), luma(b)
    pa, pb = [ya], [yb]
    for _ in range(levels - 1):
        pa.append(F.avg_pool2d(pa[-1], 2, ceil_mode=True)); pb.append(F.avg_pool2d(pb[-1], 2, ceil_mode=True))
    flow = None
    stop = int(round(math.log2(out_scale)))
    for lvl in range(levels - 1, stop - 1, -1):
        A, B = _blur(pa[lvl], 1.0), _blur(pb[lvl], 1.0)
        if flow is None:
            flow = torch.zeros_like(A).repeat(1, 2, 1, 1)
        else:
            flow = F.interpolate(flow, size=A.shape[-2:], mode='bilinear', align_corners=False) * 2.0
        for _ in range(iters):
            Bw = _warp(B, flow)
            ix, iy = _grad(0.5 * (A + Bw))
            it = Bw - A
            sxx, sxy, syy = _blur(ix * ix, sigma), _blur(ix * iy, sigma), _blur(iy * iy, sigma)
            sxt, syt = _blur(ix * it, sigma), _blur(iy * it, sigma)
            det = sxx * syy - sxy * sxy + lam * (sxx + syy) + lam * lam
            du = -((syy + lam) * sxt - sxy * syt) / det
            dv = -((sxx + lam) * syt - sxy * sxt) / det
            flow = flow + torch.cat([du, dv], 1).clamp(-2.0, 2.0)
    return flow
