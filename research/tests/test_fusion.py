"""Recoverability-guided fusion: weights, limits, and no reference access."""
import numpy as np
from research.lacunabench.fusion import disagreement, fuse, pixel_features, pooled_error


def stack(seed=0, k=3, h=90, w=160):
    return np.random.default_rng(seed).integers(0, 256, (k, h, w, 3), dtype=np.uint8)


def test_low_temperature_selects_predicted_best_candidate():
    s = stack()
    pred = np.full((3, 45, 80), 10.0)
    pred[1] = 0.0                                   # candidate 1 predicted perfect everywhere
    np.testing.assert_array_equal(fuse(s, pred, 1e-3), s[1])


def test_high_temperature_approaches_mean():
    s = stack(1)
    out = fuse(s, np.random.default_rng(2).random((3, 45, 80)), 1e6)
    assert np.abs(out.astype(float) - s.astype(float).mean(0)).max() <= 1


def test_identical_candidates_fuse_to_themselves_and_have_zero_disagreement():
    one = stack(3, k=1)[0]
    s = np.stack([one, one, one])
    np.testing.assert_array_equal(fuse(s, np.random.default_rng(4).random((3, 45, 80)) * 50, 2.0), one)
    assert np.allclose(disagreement(s), 0)


def test_shapes_and_pooled_error():
    s = stack(5, k=5, h=360, w=640)
    d = disagreement(s)
    assert d.shape == (6, 45, 80)
    rows = pixel_features(np.zeros((9, 45, 80), np.float32), d)
    assert rows.shape == (45 * 80, 2 * 15 + 15)
    assert pooled_error(s[0], s[0]).max() == 0
