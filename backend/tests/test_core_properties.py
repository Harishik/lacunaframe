"""Property-based checks of the corruption / reconstruction / evaluation contract."""
import numpy as np
from hypothesis import given, settings, strategies as st
from backend.core import corrupt, evaluate, restore

CLASSICAL = ('hold', 'linear', 'flow', 'telea', 'flow_quadratic')
videos = st.builds(
    lambda seed, n, h, w: np.random.default_rng(seed).integers(0, 256, (n, h, w, 3), dtype=np.uint8),
    st.integers(0, 10_000), st.integers(3, 9), st.integers(8, 24), st.integers(8, 24))
configs = st.fixed_dictionaries(dict(
    seed=st.integers(0, 10_000), drop=st.floats(0, .9), start=st.floats(0, 1), gap=st.integers(0, 6),
    occlusion=st.sampled_from([0, .2, .4]), noise=st.sampled_from([0, 5]), blur=st.sampled_from([0, 1.0])))
SETTINGS = settings(max_examples=40, deadline=None)


@SETTINGS
@given(videos, configs)
def test_corruption_is_deterministic_and_keeps_an_anchor(ref, cfg):
    a, b = corrupt(ref, cfg), corrupt(ref, cfg)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    damaged, missing, mask = a
    assert damaged.shape == ref.shape and damaged.dtype == np.uint8
    assert not missing.all() and mask.shape == ref.shape[:3]
    assert (damaged[missing] == 0).all()


@SETTINGS
@given(videos, configs, st.sampled_from(CLASSICAL))
def test_observed_frames_survive_and_placeholders_are_ignored(ref, cfg, method):
    damaged, missing, mask = corrupt(ref, cfg)
    out, _, unreliability = restore(damaged, missing, mask, method)
    keep = ~missing
    if method == 'telea':  # Telea may only change masked pixels of observed frames
        unchanged = ~mask[keep]
        np.testing.assert_array_equal(out[keep][unchanged], damaged[keep][unchanged])
    else:
        np.testing.assert_array_equal(out[keep], damaged[keep])
    poisoned = damaged.copy()
    poisoned[missing] = 255  # a solver must never read missing storage
    np.testing.assert_array_equal(restore(poisoned, missing, mask, method)[0], out)
    assert all(0 <= u <= 1 for u in unreliability)


@SETTINGS
@given(videos)
def test_evaluation_of_identity_is_perfect(ref):
    affected = np.zeros(ref.shape[:3], bool)
    affected[0] = True
    report = evaluate(ref, ref.copy(), affected, 12.)
    assert report['psnr'] == 'Infinity' and report['affected_psnr'] == 'Infinity'
    assert report['temporal_error'] == 0


@SETTINGS
@given(videos, st.integers(0, 10_000))
def test_affected_score_ignores_unaffected_pixels(ref, seed):
    affected = np.zeros(ref.shape[:3], bool)
    affected[:, : ref.shape[1] // 2] = True
    result = ref.copy()
    rng = np.random.default_rng(seed)
    result[~affected] = rng.integers(0, 256, result[~affected].shape, dtype=np.uint8)
    assert evaluate(ref, result, affected, 12.)['affected_psnr'] == 'Infinity'


@SETTINGS
@given(st.floats(-5, 5), st.floats(-2, 2), st.integers(1, 8), st.integers(1, 8), st.floats(0, 1))
def test_quadratic_model_recovers_constant_acceleration(v, a, s_next, s_prev, frac):
    from backend.core import quadratic_displacement
    d = lambda tau: v * tau + a * tau ** 2 / 2
    f_next, f_prev = np.array([d(s_next)]), np.array([d(-s_prev)])
    tau = frac * s_next
    assert np.allclose(quadratic_displacement(f_next, f_prev, s_next, s_prev, tau), d(tau), atol=1e-6)


@SETTINGS
@given(st.floats(-5, 5), st.integers(1, 8), st.integers(1, 8), st.floats(0, 1))
def test_quadratic_model_reduces_to_linear_for_uniform_motion(v, s_next, s_prev, frac):
    from backend.core import quadratic_displacement
    tau = frac * s_next
    out = quadratic_displacement(np.array([v * s_next]), np.array([-v * s_prev]), s_next, s_prev, tau)
    assert np.allclose(out, v * tau, atol=1e-6)
