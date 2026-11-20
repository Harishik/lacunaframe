"""Loss processes: determinism, rates, burst structure, codec dependency."""
import numpy as np
import pytest
from research.lacunabench import losses


def test_contiguous_and_protection():
    m = losses.contiguous(48, 8, 32)
    assert m.sum() == 32 and not m[:8].any() and not m[40:].any()
    with pytest.raises(ValueError):
        losses.contiguous(48, 1, 4)


def test_bernoulli_rate_and_determinism():
    a = losses.bernoulli(200_000, .25, np.random.default_rng(1))
    b = losses.bernoulli(200_000, .25, np.random.default_rng(1))
    assert (a == b).all() and abs(a.mean() - .25) < .005


def test_gilbert_elliott_matches_rate_and_mean_burst():
    m = losses.gilbert_elliott(400_000, .25, 4, np.random.default_rng(2))
    assert abs(m.mean() - .25) < .01
    assert abs(np.mean(losses.runs(m)) - 4) < .15
    bern = losses.bernoulli(400_000, .25, np.random.default_rng(2))
    assert np.mean(losses.runs(m)) > 2 * np.mean(losses.runs(bern))


def test_gop_chain_loses_until_next_iframe_and_hits_rate():
    gop = 8
    q = losses.gop_frame_loss_for_rate(.25, gop)
    m = losses.gop_chain(400_000, q, gop, np.random.default_rng(3), protect=0)
    assert abs(m.mean() - .25) < .01
    for start in range(0, 4000, gop):
        block = m[start:start + gop]
        if block.any():
            first = int(np.argmax(block))
            assert block[first:].all()  # dependency: everything after the first loss is lost


def test_protection_keeps_boundary_anchors():
    for proc in (lambda r: losses.bernoulli(48, .9, r), lambda r: losses.gilbert_elliott(48, .5, 6, r)):
        m = proc(np.random.default_rng(4))
        assert not m[:2].any() and not m[-2:].any()
