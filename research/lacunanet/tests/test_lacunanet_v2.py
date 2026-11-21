"""LacunaNet v2: init identity, trajectory maths, reversal, masking, export parity."""
import numpy as np
import torch
from research.lacunanet.model import make_cond
from research.lacunanet.model_v2 import LacunaNetV2, correlation, reverse, trajectory, anchor_loss, warp


def x4(h=64, w=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(1, 3, h, w, generator=g) for _ in range(4)]


def test_untrained_v2_reproduces_a_static_scene():
    """With the soft-argmax initialisation the untrained model already matches; for identical
    anchors the best match is zero displacement, so the output must equal the input."""
    torch.manual_seed(0)
    m = LacunaNetV2(16).eval()
    import torch.nn.functional as F
    a = F.avg_pool2d(torch.rand(1, 3, 368, 640), 3, 1, 1)   # realistic size; untrained features only bias corners
    for t in (.25, .5, .75):
        with torch.no_grad():
            f, s = m(a, a, a, a, torch.tensor([make_cond(t, 3, -t - .5, 1 - t + .5, True, True)]))
        assert (f - a).abs().mean() < 0.02


def test_trajectory_matches_constant_acceleration():
    v, acc = torch.tensor(1.5), torch.tensor(-0.8)
    d = lambda tau: v * tau + acc * tau ** 2 / 2
    s_prev = torch.tensor([0.5]); one = torch.ones(1)
    f_next, f_prev = d(1.0).expand(1, 2, 4, 4), d(-0.5).expand(1, 2, 4, 4)
    out = trajectory(f_next, f_prev, one, s_prev, torch.tensor([0.3]), torch.ones(1))
    assert torch.allclose(out, d(0.3).expand_as(out), atol=1e-5)
    lin = trajectory(f_next, f_prev, one, s_prev, torch.tensor([0.3]), torch.zeros(1))  # no outer anchor
    assert torch.allclose(lin, (f_next * 0.3), atol=1e-6)


def test_reversal_inverts_smooth_translation():
    disp = torch.zeros(1, 2, 32, 48); disp[:, 0] = 2.5; disp[:, 1] = -1.0
    b = reverse(disp)
    assert torch.allclose(b, -disp, atol=1e-4)


def test_correlation_peaks_at_true_shift():
    a = torch.randn(1, 8, 20, 20)
    b = torch.roll(a, shifts=(1, 2), dims=(2, 3))   # b(p + (1,2)) = a(p)
    c = correlation(a, b, 3)[0, :, 5:15, 5:15].mean((1, 2))
    assert int(c.argmax()) == (3 + 1) * 7 + (3 + 2)


def test_absent_outer_anchors_cannot_leak():
    torch.manual_seed(1)
    m = LacunaNetV2(16).eval()
    for p in m.parameters():
        p.data += 0.05 * torch.randn_like(p)
    a, b, c, d = x4(seed=2)
    cond = torch.tensor([make_cond(.4, 2, -1.0, 1.0, False, False)])
    with torch.no_grad():
        f1, _ = m(a, b, torch.zeros_like(c), torch.zeros_like(d), cond)
        f2, _ = m(a, b, torch.rand_like(c), torch.rand_like(d), cond)
    assert torch.allclose(f1, f2, atol=1e-5)


def test_anchor_loss_is_zero_for_identical_anchors_and_zero_flow():
    a = torch.rand(1, 3, 32, 32); z = torch.zeros(1, 2, 32, 32)
    assert float(anchor_loss(a, a, z, z)) < 1e-2


def test_onnx_export_matches_torch(tmp_path):
    import onnxruntime as ort
    from research.lacunanet.export import export
    torch.manual_seed(3)
    m = LacunaNetV2(16).eval()
    for p in m.parameters():
        p.data += 0.05 * torch.randn_like(p)
    path = tmp_path / 'v2.onnx'
    export(m, path, 64, 96)
    s = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    x = x4(368, 640, seed=4)
    cond = torch.tensor([make_cond(.3, 5, -1.4, 1.3, True, True)])
    with torch.no_grad():
        ref = m(*x, cond)
    got = s.run(None, dict(zip(['a_m1', 'a_p1', 'a_m2', 'a_p2', 'cond'], [v.numpy() for v in (*x, cond)])))
    for r, g in zip(ref, got):
        assert np.abs(r.numpy() - g).max() < 1e-4


def test_a1_side_uses_backward_flow_when_next_anchor_absent():
    """Regression test: without Ap, A1 -> target motion must be f10 * (1 - t), not zero."""
    torch.manual_seed(0)
    m = LacunaNetV2(16).eval()
    shift = 4.0
    with torch.no_grad():
        # Force the pair estimator to return a known constant flow by monkeypatching.
        m.pair.forward = lambda ea, eb, levels=3, all_levels=False, imgs=None: [torch.full_like(ea[1][:, :2], shift)] if all_levels else torch.full_like(ea[1][:, :2], shift)
        a, b, c, d = x4(64, 96)
        t = 0.25
        b0, b1, *_ = m.motion(a, b, torch.zeros_like(c), torch.zeros_like(d), torch.tensor([make_cond(t, 1, 0, 0, False, False)]))
    # d0 = shift * t, d1 = shift * (1 - t): reversed (backward) flows are their negatives at quarter resolution
    assert torch.allclose(b0, torch.full_like(b0, -shift * t), atol=1e-3)
    assert torch.allclose(b1, torch.full_like(b1, -shift * (1 - t)), atol=1e-3)


def test_v3_prior_recovers_known_shift_untrained():
    """v3 (Lucas-Kanade prior + zero-initialised residual heads) recovers a real-texture shift at step 0."""
    import torch.nn.functional as F
    from research.lacunanet.classical_flow import _warp
    torch.manual_seed(0)
    m = LacunaNetV2(16, prior='lk').eval()
    g = torch.Generator().manual_seed(1)
    a = F.avg_pool2d(torch.rand(1, 3, 128 + 8, 192 + 8, generator=g), 9, 1)   # smooth random texture
    a = (a - a.amin()) / (a.amax() - a.amin())
    shift = torch.zeros(1, 2, 128, 192); shift[:, 0] = 6.0; shift[:, 1] = -3.0
    b = _warp(a, -shift)                                                    # a(p) = b(p + shift)
    with torch.no_grad():
        e0, e1 = m.enc(a), m.enc(b)
        f = m.pair(e0, e1, imgs=(a, b)) * 4                                  # quarter grid -> full-res px
    inner = f[:, :, 4:-4, 4:-4].mean((2, 3))[0]
    assert abs(inner[0].item() - 6.0) < 0.3 and abs(inner[1].item() + 3.0) < 0.3, inner


def test_v3_identical_anchors_give_linear_blend_exactly():
    torch.manual_seed(0)
    m = LacunaNetV2(16, prior='lk').eval()
    a = torch.rand(1, 3, 64, 96)
    z = torch.zeros_like(a)
    with torch.no_grad():
        out, _ = m(a, a, z, z, torch.tensor([make_cond(0.5, 1, 0, 0, False, False)]))
    assert (out - a).abs().max().item() < 1e-3
