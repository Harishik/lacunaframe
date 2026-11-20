"""LacunaNet: multi-anchor, gap-aware missing-frame reconstruction with a built-in self-error head.

Designed for CPU inference (depthwise-separable convolutions, three pyramid levels,
grid_sample warping, ONNX-exportable). See research/protocols/RECOVERABILITY_V4.md.

Inputs (all float32):
  a_m1, a_p1     inner anchors (B, 3, H, W) in [0, 1], always present
  a_m2, a_p2     outer anchors (B, 3, H, W) in [0, 1], zeros when absent
  cond           (B, 8): [t, log2(gap)/5, tau_m2, tau_m1, tau_p1, tau_p2, mask_m2, mask_p2]
                 tau_* are anchor times relative to the target, in units of the inner span
Outputs:
  frame          (B, 3, H, W) reconstruction in [0, 1]
  scale          (B, 1, H, W) Laplace scale b > 0 of the expected per-pixel |error| (mean over RGB)
H and W must be multiples of 8.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

COND = 8


def act() -> nn.Module:
    return nn.LeakyReLU(0.1, inplace=True)


class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 with a residual connection when shapes match."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1)
        self.act = act()
        self.res = stride == 1 and cin == cout

    def forward(self, x):
        y = self.act(self.pw(self.dw(x)))
        return x + y if self.res else y


class Encoder(nn.Module):
    """Shared three-level pyramid: features at 1/2, 1/4, 1/8 resolution."""

    def __init__(self, c: int):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv2d(3, c, 3, 2, 1), act(), DSConv(c, c))
        self.l2 = nn.Sequential(DSConv(c, 2 * c, 2), DSConv(2 * c, 2 * c))
        self.l3 = nn.Sequential(DSConv(2 * c, 4 * c, 2), DSConv(4 * c, 4 * c))

    def forward(self, x):
        f1 = self.l1(x)
        f2 = self.l2(f1)
        return f1, f2, self.l3(f2)


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warp: sample x at p + flow(p). flow is (B, 2, H, W) in pixels of x."""
    b, _, h, w = x.shape
    ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, h, 1).expand(b, h, w)
    xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, w).expand(b, h, w)
    gx = xs + flow[:, 0] * (2.0 / max(w - 1, 1))
    gy = ys + flow[:, 1] * (2.0 / max(h - 1, 1))
    return F.grid_sample(x, torch.stack([gx, gy], dim=-1), mode='bilinear', padding_mode='border', align_corners=True)


class MotionLevel(nn.Module):
    """One coarse-to-fine level: update target->inner flows and the occlusion logit, FiLM-conditioned."""

    def __init__(self, cf: int, hidden: int, first: bool):
        super().__init__()
        cin = 4 * cf + (0 if first else 5)          # 4 anchor features (+ previous flows and mask)
        self.inp = nn.Conv2d(cin, hidden, 1)          # 1x1 projection; spatial context comes from the DS blocks (v4 amendment 1)
        self.film = nn.Linear(COND, 2 * hidden)
        self.body = nn.Sequential(DSConv(hidden, hidden), DSConv(hidden, hidden), DSConv(hidden, hidden))
        self.out = nn.Conv2d(hidden, 5, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.first = first

    def forward(self, feats, cond, prev):
        x = torch.cat(feats if self.first else feats + [prev], dim=1)
        h = self.inp(x)
        gamma, beta = self.film(cond).chunk(2, dim=1)
        h = act()(h * (1 + gamma[..., None, None]) + beta[..., None, None])
        return self.out(self.body(h))


class LacunaNet(nn.Module):
    def __init__(self, c: int = 32):
        super().__init__()
        self.c = c
        self.enc = Encoder(c)
        self.lv3 = MotionLevel(4 * c, 4 * c, first=True)
        self.lv2 = MotionLevel(2 * c, 3 * c, first=False)
        self.lv1 = MotionLevel(c, 2 * c, first=False)
        rc = 2 * c
        # Refinement runs at 1/2 resolution (v4 amendment 1): full-resolution evidence enters through
        # space-to-depth (pixel unshuffle) and leaves through pixel-shuffle heads, so no dense
        # convolution runs at full resolution.
        self.unshuffle, self.shuffle = nn.PixelUnshuffle(2), nn.PixelShuffle(2)
        self.ref_in = nn.Conv2d(4 * (3 * 3 + 1) + c, rc, 3, 1, 1)
        self.ref_film = nn.Linear(COND, 2 * rc)
        self.ref_down = DSConv(rc, 2 * rc, 2)
        self.ref_mid = nn.Sequential(DSConv(2 * rc, 2 * rc), DSConv(2 * rc, 2 * rc))
        self.ref_up = nn.Conv2d(2 * rc + rc, rc, 3, 1, 1)
        self.ref_body = DSConv(rc, rc)
        self.ref_out = nn.Conv2d(rc, 4 * 3, 3, 1, 1)
        self.err_out = nn.Conv2d(rc, 4 * 1, 3, 1, 1)
        nn.init.zeros_(self.ref_out.weight)
        nn.init.zeros_(self.ref_out.bias)

    @staticmethod
    def _outer_flow(inner_flow, tau_inner, tau_outer):
        """Linear extrapolation of a target->inner flow to an outer anchor (initial guess for context warping)."""
        ratio = (tau_outer / torch.where(tau_inner.abs() < 1e-6, torch.full_like(tau_inner, 1e-6), tau_inner))
        return inner_flow * ratio[:, None, None, None]

    def _level_feats(self, level, fm1, fp1, fm2, fp2, flows, cond):
        tau_m2, tau_m1, tau_p1, tau_p2 = cond[:, 2], cond[:, 3], cond[:, 4], cond[:, 5]
        mask_m2, mask_p2 = cond[:, 6][:, None, None, None], cond[:, 7][:, None, None, None]
        if flows is None:
            return [fm1, fp1, fm2 * mask_m2, fp2 * mask_p2]
        fl_m, fl_p = flows[:, 0:2], flows[:, 2:4]
        return [warp(fm1, fl_m), warp(fp1, fl_p),
                warp(fm2, self._outer_flow(fl_m, tau_m1, tau_m2)) * mask_m2,
                warp(fp2, self._outer_flow(fl_p, tau_p1, tau_p2)) * mask_p2]

    def forward(self, a_m1, a_p1, a_m2, a_p2, cond):
        e = [self.enc(x) for x in (a_m1, a_p1, a_m2, a_p2)]       # 4 x (f1, f2, f3)
        t = cond[:, 0][:, None, None, None]
        # Level 3 (1/8): initial flows and mask logit.
        d = self.lv3(self._level_feats(3, e[0][2], e[1][2], e[2][2], e[3][2], None, cond), cond, None)
        state = d                                                  # [flow_m(2), flow_p(2), mask(1)] at 1/8
        for level, module, idx in ((2, self.lv2, 1), (1, self.lv1, 0)):
            state = F.interpolate(state, scale_factor=2.0, mode='bilinear', align_corners=False)
            state = torch.cat([state[:, :4] * 2.0, state[:, 4:]], dim=1)
            feats = self._level_feats(level, e[0][idx], e[1][idx], e[2][idx], e[3][idx], state, cond)
            state = state + module(feats, cond, state)
        state = F.interpolate(state, scale_factor=2.0, mode='bilinear', align_corners=False)
        flow_m, flow_p, logit = state[:, 0:2] * 2.0, state[:, 2:4] * 2.0, state[:, 4:5]
        wm, wp = warp(a_m1, flow_m), warp(a_p1, flow_p)
        # Occlusion-aware blend, biased by temporal proximity to each inner anchor.
        mask = torch.sigmoid(logit + torch.log((1 - t + 1e-3) / (t + 1e-3)))
        base = mask * wm + (1 - mask) * wp
        f1 = e[0][0] * 0.5 + e[1][0] * 0.5                               # 1/2 resolution
        h = self.ref_in(torch.cat([self.unshuffle(torch.cat([base, wm, wp, mask], dim=1)), f1], dim=1))
        gamma, beta = self.ref_film(cond).chunk(2, dim=1)
        h = act()(h * (1 + gamma[..., None, None]) + beta[..., None, None])
        m = self.ref_mid(self.ref_down(h))
        h = act()(self.ref_up(torch.cat([F.interpolate(m, scale_factor=2.0, mode='bilinear', align_corners=False), h], dim=1)))
        h = self.ref_body(h)
        frame = torch.clamp(base + self.shuffle(self.ref_out(h)), 0.0, 1.0)
        scale = F.softplus(self.shuffle(self.err_out(h))) + 1e-3
        return frame, scale


def make_cond(t: float, gap: int, tau_m2: float, tau_p2: float, has_m2: bool, has_p2: bool) -> list[float]:
    """Conditioning vector; tau_m1 = -t and tau_p1 = 1 - t by construction (inner-span units)."""
    import math
    clip = lambda v: max(-4.0, min(4.0, v))
    return [t, math.log2(max(gap, 1)) / 5.0, clip(tau_m2) if has_m2 else 0.0, -t, 1.0 - t,
            clip(tau_p2) if has_p2 else 0.0, float(has_m2), float(has_p2)]


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
