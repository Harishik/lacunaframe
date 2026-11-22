"""LacunaNet v2: matching pyramid + physics trajectory + differentiable flow reversal.

See research/lacunanet/DESIGN_AUDIT.md for the evidence behind each component.
Interface is identical to v1 (same inputs/outputs, same 8-value conditioning), so the
runtime adapter, export and evaluation code are unchanged.

Notation: inner anchors A0 (time 0) and A1 (time 1, inner-span units), target time t,
optional outer anchors Am at -s_m and Ap at 1 + s_p.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .model import COND, DSConv, act, make_cond, parameter_count, warp  # noqa: F401  (re-exported)
    from .classical_flow import lk_flow
except ImportError:   # flat layout (Colab: files side by side)
    from model import COND, DSConv, act, make_cond, parameter_count, warp  # noqa: F401
    from classical_flow import lk_flow

RADIUS = 3
PRIOR = 0.5   # logit penalty per squared pixel of displacement in the coarse soft-argmax


class Encoder4(nn.Module):
    """Shared pyramid at 1/2, 1/4, 1/8, 1/16."""

    def __init__(self, c: int):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv2d(3, c, 3, 2, 1), act(), DSConv(c, c))
        self.l2 = nn.Sequential(DSConv(c, 2 * c, 2), DSConv(2 * c, 2 * c))
        self.l3 = nn.Sequential(DSConv(2 * c, 3 * c, 2), DSConv(3 * c, 3 * c))
        self.l4 = nn.Sequential(DSConv(3 * c, 4 * c, 2), DSConv(4 * c, 4 * c))

    def forward(self, x):
        f1 = self.l1(x); f2 = self.l2(f1); f3 = self.l3(f2)
        return f1, f2, f3, self.l4(f3)


def correlation(a: torch.Tensor, b: torch.Tensor, r: int = RADIUS) -> torch.Tensor:
    """Local cost volume: normalised dot products of a(p) with b(p + d), |d|_inf <= r. ((2r+1)^2 channels)."""
    # Pearson correlation (centre each feature vector, then normalise). Post-activation features are
    # mostly positive, so raw cosine similarity is high everywhere and cannot separate the true match.
    a = F.normalize(a - a.mean(1, keepdim=True), dim=1)
    b = F.normalize(b - b.mean(1, keepdim=True), dim=1)
    h, w = a.shape[-2:]
    bp = F.pad(b, [r, r, r, r], mode='replicate')   # zero padding biases border matches inward
    out = [(a * bp[:, :, dy:dy + h, dx:dx + w]).sum(1, keepdim=True) for dy in range(2 * r + 1) for dx in range(2 * r + 1)]
    return torch.cat(out, dim=1)


class FlowHead(nn.Module):
    def __init__(self, cf: int, hidden: int, gated: bool = False):
        super().__init__()
        self.inp = nn.Conv2d(cf + (2 * RADIUS + 1) ** 2 + 2, hidden, 1)
        self.body = nn.Sequential(DSConv(hidden, hidden), DSConv(hidden, hidden), DSConv(hidden, hidden))
        self.out = nn.Conv2d(hidden, 2, 3, 1, 1)
        if gated:
            # ReZero gate: normally initialised output scaled by a scalar starting at 0. The gate's gradient
            # pools evidence over every pixel, so the correction enters only as it lowers the loss on average,
            # instead of Adam's sign-like first steps on a zero-initialised conv spraying offsets onto the prior.
            self.gain = nn.Parameter(torch.zeros(1))
        else:
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
            self.gain = None

    def forward(self, fa, corr, flow):
        out = self.out(self.body(act()(self.inp(torch.cat([fa, corr, flow], 1)))))
        return out if self.gain is None else self.gain * out


def offset_validity(ref: torch.Tensor, r: int = RADIUS) -> torch.Tensor:
    """1 where p + offset lies inside the image, 0 otherwise ((2r+1)^2 channels, same order as correlation)."""
    ones = torch.ones_like(ref[:, :1])
    h, w = ref.shape[-2:]
    op = F.pad(ones, [r, r, r, r])
    return torch.cat([op[:, :, dy:dy + h, dx:dx + w] for dy in range(2 * r + 1) for dx in range(2 * r + 1)], dim=1)


def soft_argmax(corr: torch.Tensor, temperature: float = 0.05, r: int = RADIUS, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable expected displacement under softmax(corr / T) over in-image offsets only."""
    k = 2 * r + 1
    offs = torch.arange(-r, r + 1, dtype=corr.dtype, device=corr.device)
    dy = offs.view(k, 1).expand(k, k).reshape(1, k * k, 1, 1)
    dx = offs.view(1, k).expand(k, k).reshape(1, k * k, 1, 1)
    # Weak zero-motion prior resolves ambiguous (textureless / aperture) matches toward no motion;
    # a distinctive match (correlation margin >~ 0.1) easily overrides it.
    logits = corr / temperature - PRIOR * (dx ** 2 + dy ** 2)
    if valid is not None:
        logits = logits * valid - 1e4 * (1 - valid)   # out-of-image offsets get no weight
    w = torch.softmax(logits, dim=1)
    return torch.cat([(w * dx).sum(1, keepdim=True), (w * dy).sum(1, keepdim=True)], 1)


def _pool_flow(flow: torch.Tensor, ratio: int) -> torch.Tensor:
    """Average-pool a flow field by a fixed integer ratio (static kernel: exportable) and rescale its units."""
    return flow if ratio == 1 else F.avg_pool2d(flow, ratio) / ratio


class PairFlow(nn.Module):
    """Coarse-to-fine forward displacement a -> b on a's grid (PWC-style), shared by all anchor pairs.

    prior=None (v2 as designed): the coarsest level starts from a soft-argmax over the cost volume.
    prior='lk' (v3): every level starts from a parameter-free, differentiable pyramidal Lucas-Kanade
    flow (classical_flow.lk_flow), and the learned heads predict a residual on top of it from the
    cost volume around the prior. Heads are zero-initialised, so at step 0 the motion is exactly the
    classical estimate; learning only has to fix where the classical model fails (large motion,
    occlusion, aperture). Every level's flow is returned so each can be supervised."""

    def __init__(self, c: int, prior: str | None = None):
        super().__init__()
        self.prior = prior
        g = prior is not None
        self.heads = nn.ModuleList([FlowHead(4 * c, 3 * c, g), FlowHead(3 * c, 2 * c, g), FlowHead(2 * c, 2 * c, g)])  # 1/16, 1/8, 1/4
        self.temperature = nn.Parameter(torch.tensor(0.02))

    def forward(self, ea, eb, levels: int = 3, all_levels: bool = False, imgs=None):
        """ea/eb: encoder outputs (f1, f2, f3, f4); imgs=(a, b) full-resolution images (needed for prior='lk').
        Returns the finest flow (in that level's pixels), or the per-level list when all_levels."""
        if self.prior == 'lk':
            base_scale = 4 if levels >= 3 else 8
            with torch.no_grad():   # parameter-free; inputs need no gradient
                base = lk_flow(imgs[0], imgs[1], out_scale=base_scale)
        flow, resid, flows = None, None, []
        for k, (idx, head) in enumerate(zip((3, 2, 1), self.heads)):
            if k >= levels:
                break
            fa, fb = ea[idx], eb[idx]
            if self.prior == 'lk':
                prior = _pool_flow(base, (16, 8, 4)[k] // base_scale)
                resid = torch.zeros_like(prior) if resid is None else F.interpolate(resid, size=fa.shape[-2:], mode='bilinear', align_corners=False) * 2.0
                cur = prior + resid
                resid = resid + head(fa, correlation(fa, warp(fb, cur)), cur)
                flow = prior + resid
            else:
                if flow is None:
                    flow = soft_argmax(correlation(fa, fb), self.temperature.abs() + 1e-3, valid=offset_validity(fa))
                    corr = correlation(fa, fb)
                else:
                    flow = F.interpolate(flow, size=fa.shape[-2:], mode='bilinear', align_corners=False) * 2.0
                    corr = correlation(fa, warp(fb, flow))
                flow = flow + head(fa, corr, flow)
            flows.append(flow)
        return flows if all_levels else flow


def _to_quarter(flow: torch.Tensor, size) -> torch.Tensor:
    scale = size[0] / flow.shape[-2]
    return F.interpolate(flow, size=size, mode='bilinear', align_corners=False) * scale


def trajectory(f_next, f_prev, s_next, s_prev, tau, has_prev):
    """Constant-acceleration displacement at time tau from an anchor whose neighbours sit at +s_next
    (displacement f_next) and -s_prev (displacement f_prev). Without the previous neighbour: constant velocity."""
    s_next = s_next[:, None, None, None]; s_prev = s_prev.clamp(min=1e-3)[:, None, None, None]
    tau = tau[:, None, None, None]; has_prev = has_prev[:, None, None, None]
    accel = 2 * (f_next * s_prev + f_prev * s_next) / (s_prev * s_next * (s_prev + s_next)) * has_prev
    velocity = (f_next - accel * s_next ** 2 / 2) / s_next
    return velocity * tau + accel * tau ** 2 / 2


def reverse(disp: torch.Tensor, iters: int = 4) -> torch.Tensor:
    """Backward flow B with q + B(q) = p where p + disp(p) = q (fixed-point inversion, differentiable)."""
    b = -disp
    for _ in range(iters):
        b = -warp(disp, b)
    return b


class LacunaNetV2(nn.Module):
    def __init__(self, c: int = 24, prior: str | None = None):
        super().__init__()
        self.c, self.prior = c, prior
        self.enc = Encoder4(c)
        self.pair = PairFlow(c, prior)
        h = 2 * c
        self.cor_in = nn.Conv2d(2 * (2 * c) + 4 + 1, h, 1)
        self.cor_film = nn.Linear(COND, 2 * h)
        self.cor_body = nn.Sequential(DSConv(h, h), DSConv(h, h), DSConv(h, h))
        self.cor_out = nn.Conv2d(h, 5, 3, 1, 1)
        self.gated = prior is not None
        if self.gated:   # ReZero gates (see FlowHead): flow correction, blend logit and image residual
            self.cor_gain = nn.Parameter(torch.zeros(5))
        else:
            nn.init.zeros_(self.cor_out.weight); nn.init.zeros_(self.cor_out.bias)
        rc = 2 * c
        self.unshuffle, self.shuffle = nn.PixelUnshuffle(2), nn.PixelShuffle(2)
        self.ref_in = nn.Conv2d(4 * (3 * 3 + 1) + c, rc, 3, 1, 1)
        self.ref_film = nn.Linear(COND, 2 * rc)
        self.ref_down = DSConv(rc, 2 * rc, 2)
        self.ref_mid = nn.Sequential(DSConv(2 * rc, 2 * rc), DSConv(2 * rc, 2 * rc))
        self.ref_up = nn.Conv2d(2 * rc + rc, rc, 3, 1, 1)
        self.ref_body = DSConv(rc, rc)
        self.ref_out = nn.Conv2d(rc, 12, 3, 1, 1)
        self.err_out = nn.Conv2d(rc, 4, 3, 1, 1)
        if self.gated:
            self.ref_gain = nn.Parameter(torch.zeros(1))
        else:
            nn.init.zeros_(self.ref_out.weight); nn.init.zeros_(self.ref_out.bias)

    def motion(self, a0, a1, am, ap, cond):
        """Returns quarter-resolution backward flows B0, B1 (target -> A0/A1), blend logit, pair flows and encodings."""
        e0, e1, em, ep = (self.enc(x) for x in (a0, a1, am, ap))
        t, tau_m2, tau_p2, has_m, has_p = cond[:, 0], cond[:, 2], cond[:, 5], cond[:, 6], cond[:, 7]
        s_m = (-tau_m2 - t) * has_m + (1 - has_m)          # outer spacing in inner-span units (1 when absent)
        s_p = (tau_p2 - (1 - t)) * has_p + (1 - has_p)
        q = e0[1].shape[-2:]
        p01, p10 = self.pair(e0, e1, all_levels=True, imgs=(a0, a1)), self.pair(e1, e0, all_levels=True, imgs=(a1, a0))
        f01, f10 = p01[-1], p10[-1]                                         # 1/4, full detail
        f0m = _to_quarter(self.pair(e0, em, levels=2, imgs=(a0, am)), q) * has_m[:, None, None, None]   # coarse: acceleration is smooth
        f1p = _to_quarter(self.pair(e1, ep, levels=2, imgs=(a1, ap)), q) * has_p[:, None, None, None]
        one = torch.ones_like(t)
        d0 = trajectory(f01, f0m, one, s_m, t, has_m)                      # A0 -> target displacement on A0 grid
        # A1 side: the always-present neighbour is A0 (at tau = -1, displacement f10); the optional one is Ap.
        # With Ap: constant acceleration through both. Without: constant velocity from f10, i.e. f10 * (1 - t).
        hp = has_p[:, None, None, None]
        d1 = hp * trajectory(f1p, f10, s_p, one, t - 1, torch.ones_like(t)) + (1 - hp) * f10 * (1 - t)[:, None, None, None]
        b0, b1 = reverse(d0), reverse(d1)
        incons = (f01 + warp(f10, f01)).norm(dim=1, keepdim=True)
        x = torch.cat([warp(e0[1], b0), warp(e1[1], b1), b0, b1, warp(incons, b0)], 1)
        hdn = self.cor_in(x)
        gamma, beta = self.cor_film(cond).chunk(2, dim=1)
        hdn = act()(hdn * (1 + gamma[..., None, None]) + beta[..., None, None])
        delta = self.cor_out(self.cor_body(hdn))
        if self.gated:
            delta = delta * self.cor_gain.view(1, 5, 1, 1)
        self._pyramid = (p01, p10)
        return b0 + delta[:, 0:2], b1 + delta[:, 2:4], delta[:, 4:5], f01, f10, e0, e1

    def forward(self, a_m1, a_p1, a_m2, a_p2, cond, return_aux: bool = False):
        b0, b1, logit, f01, f10, e0, e1 = self.motion(a_m1, a_p1, a_m2, a_p2, cond)
        size = a_m1.shape[-2:]
        up = lambda f: F.interpolate(f, size=size, mode='bilinear', align_corners=False) * (size[0] / f.shape[-2])
        b0f, b1f = up(b0), up(b1)
        logit = F.interpolate(logit, size=size, mode='bilinear', align_corners=False)
        t = cond[:, 0][:, None, None, None]
        w0, w1 = warp(a_m1, b0f), warp(a_p1, b1f)
        mask = torch.sigmoid(logit + torch.log((1 - t + 1e-3) / (t + 1e-3)))
        base = mask * w0 + (1 - mask) * w1
        # Context features aligned to the target time with the same flows and occlusion mask as the pixels.
        # (v1 and the first v2 averaged *unwarped* anchor features: a ghosted, linear-blend-like signal that
        # pulled the refinement toward blending; DESIGN_AUDIT section 7.)
        half = e0[0].shape[-2:]
        hb = lambda f: F.interpolate(f, size=half, mode='bilinear', align_corners=False) * (half[0] / f.shape[-2])
        mh = F.interpolate(mask, size=half, mode='bilinear', align_corners=False)
        f1 = mh * warp(e0[0], hb(b0)) + (1 - mh) * warp(e1[0], hb(b1))
        h = self.ref_in(torch.cat([self.unshuffle(torch.cat([base, w0, w1, mask], 1)), f1], 1))
        gamma, beta = self.ref_film(cond).chunk(2, dim=1)
        h = act()(h * (1 + gamma[..., None, None]) + beta[..., None, None])
        m = self.ref_mid(self.ref_down(h))
        h = act()(self.ref_up(torch.cat([F.interpolate(m, scale_factor=2.0, mode='bilinear', align_corners=False), h], 1)))
        h = self.ref_body(h)
        res = self.shuffle(self.ref_out(h))
        frame = torch.clamp(base + (res * self.ref_gain if self.gated else res), 0.0, 1.0)
        # Stop-gradient: the self-error head describes the reconstruction, it must not reshape it.
        scale = F.softplus(self.shuffle(self.err_out(h.detach()))) + 1e-3
        if return_aux:
            return frame, scale, dict(f01=up(f01), f10=up(f10), pyramid=self._pyramid, base=base)
        return frame, scale


def anchor_loss_multiscale(a0, a1, pyramid):
    """Anchor loss at every pyramid level on area-downsampled anchors, so each level is supervised
    inside its bilinear capture range (a 12 px full-resolution shift is 0.75 px at 1/16)."""
    p01, p10 = pyramid
    total = 0.0
    for f01, f10, wgt in zip(p01, p10, (1.0, 0.5, 0.25)):
        size = f01.shape[-2:]
        d0, d1 = F.adaptive_avg_pool2d(a0, size), F.adaptive_avg_pool2d(a1, size)
        total = total + wgt * anchor_loss(d0, d1, f01, f10)
    return total


def anchor_loss_final(a0, a1, f01, f10):
    """v3 anchor loss: supervise the *final* quarter-resolution pair flows at three scales (the final flow
    average-pooled to 1/8 and 1/16 for capture range). Per-level intermediate flows get no targets of their
    own: with a fine-level prior, the coarse photometric optimum (pooling mixes motions) differs from the fine
    truth, and a coarse correction trained toward it is carried onto the precise fine prior (DESIGN_AUDIT 8)."""
    total = 0.0
    for ratio, wgt in ((1, 1.0), (2, 0.5), (4, 0.25)):
        g01, g10 = _pool_flow(f01, ratio), _pool_flow(f10, ratio)
        size = g01.shape[-2:]
        total = total + wgt * anchor_loss(F.adaptive_avg_pool2d(a0, size), F.adaptive_avg_pool2d(a1, size), g01, g10)
    return total


def anchor_loss(a0, a1, f01, f10):
    """Free motion supervision: inner anchors must warp onto each other; occlusions down-weighted by
    forward/backward inconsistency; plus first-order smoothness of the pair flows."""
    w = lambda img, fl: warp(img, fl)
    inc0 = (f01 + w(f10, f01)).norm(dim=1, keepdim=True).detach()
    inc1 = (f10 + w(f01, f10)).norm(dim=1, keepdim=True).detach()
    occ0, occ1 = torch.exp(-inc0 / 2.0), torch.exp(-inc1 / 2.0)
    ch = lambda x, y, m: (torch.sqrt((x - y) ** 2 + 1e-6).mean(1, keepdim=True) * m).sum() / (m.sum() + 1e-6)
    photo = ch(w(a1, f01), a0, occ0) + ch(w(a0, f10), a1, occ1)
    smooth = sum((f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean() + (f[:, :, 1:] - f[:, :, :-1]).abs().mean() for f in (f01, f10))
    return photo + 0.01 * smooth
