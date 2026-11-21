"""LacunaNet unit tests (training environment: ../.venv-train/bin/python -m pytest research/lacunanet/tests)."""
import random
import numpy as np
import torch
from research.lacunanet.model import LacunaNet, make_cond, warp


def anchors(h=64, w=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(1, 3, h, w, generator=g) for _ in range(4)]


def test_zero_flow_warp_is_identity():
    x = torch.rand(1, 3, 32, 48)
    assert torch.allclose(warp(x, torch.zeros(1, 2, 32, 48)), x, atol=1e-5)  # float32 grid rounding


def test_untrained_model_is_exact_temporal_linear_blend():
    torch.manual_seed(0)
    m = LacunaNet(16).eval()
    a, b, c, d = anchors()
    for t in (.25, .5, .75):
        with torch.no_grad():
            frame, scale = m(a, b, c, d, torch.tensor([make_cond(t, 3, -t - .5, 1 - t + .5, True, True)]))
        blend = (1 - t) * a + t * b
        assert torch.allclose(frame, blend, atol=2e-3), (t, (frame - blend).abs().max())
        assert (scale > 0).all()


def test_shapes_and_resolution_independence():
    m = LacunaNet(16).eval()
    for h, w in ((64, 96), (256, 448), (360, 640)):
        x = [torch.rand(2, 3, h, w) for _ in range(4)]
        cond = torch.tensor([make_cond(.3, 4, -1.3, 1.2, True, False)] * 2)
        with torch.no_grad():
            f, s = m(*x, cond)
        assert f.shape == (2, 3, h, w) and s.shape == (2, 1, h, w)


def test_outer_anchor_mask_removes_its_influence():
    torch.manual_seed(1)
    m = LacunaNet(16).eval()
    for p in m.parameters():                      # make the network non-trivial
        p.data += 0.05 * torch.randn_like(p)
    a, b, c, d = anchors(seed=2)
    cond = torch.tensor([make_cond(.4, 2, -1.0, 1.0, False, False)])
    with torch.no_grad():
        f1, _ = m(a, b, torch.zeros_like(c), torch.zeros_like(d), cond)
        f2, _ = m(a, b, torch.rand_like(c), torch.rand_like(d), cond)
    assert torch.allclose(f1, f2, atol=1e-6)      # absent outer anchors cannot leak in


def test_onnx_export_matches_torch(tmp_path):
    import onnxruntime as ort
    from research.lacunanet.export import export
    torch.manual_seed(3)
    m = LacunaNet(16).eval()
    for p in m.parameters():
        p.data += 0.05 * torch.randn_like(p)
    path = tmp_path / 'ln.onnx'
    export(m, path, 64, 96)
    s = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    x = anchors(360, 640, seed=4)
    cond = torch.tensor([make_cond(.3, 5, -1.4, 1.3, True, True)])
    with torch.no_grad():
        ref = m(*x, cond)
    got = s.run(None, dict(zip(['a_m1', 'a_p1', 'a_m2', 'a_p2', 'cond'], [v.numpy() for v in (*x, cond)])))
    for r, g in zip(ref, got):
        assert np.abs(r.numpy() - g).max() < 1e-4


def test_gopro_sampler(tmp_path):
    from research.lacunanet.data import GoproSource
    frames = np.random.default_rng(0).integers(0, 255, (40, 360, 640, 3), dtype=np.uint8)
    np.savez(tmp_path / 's.npz', frames=frames)
    src, rng = GoproSource(tmp_path), random.Random(0)
    for _ in range(50):
        crop, cond, (m2, p2) = src.sample(rng)
        assert len(crop) == 5 and all(c.shape == (256, 256, 3) for c in crop)
        assert 0 < cond[0] < 1 and abs(cond[3] + cond[0]) < 1e-9 and abs(cond[4] - (1 - cond[0])) < 1e-9
        assert cond[6] == float(m2) and cond[7] == float(p2)
