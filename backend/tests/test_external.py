"""ONNX adapters for published interpolators. Skipped when exports are not present."""
import numpy as np
import pytest
from backend import external

pytestmark = pytest.mark.skipif(not all(external.available(m) for m in ('amt_s', 'ifrnet_s', 'ifrnet')),
                                reason='baseline exports not present (see research/metrics/export_baselines.py)')


def scene(h=360, w=640, shift=6, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.random((h, w + 2 * shift, 3)).astype(np.float32), (0, 0), 4)
    base = ((base - base.min()) / (base.max() - base.min()) * 255).astype(np.uint8)
    return base[:, :w], base[:, shift:shift + w], base[:, 2 * shift:2 * shift + w]


@pytest.mark.parametrize('name', ['amt_s', 'ifrnet_s', 'ifrnet'])
def test_shape_endpoints_and_midpoint_quality(name):
    a, mid, b = scene()
    out = external.interpolate(name, a, b, .5)
    assert out.shape == a.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(external.interpolate(name, a, b, 0), a)
    linear = ((a.astype(float) + b) / 2).round()
    err = lambda x: np.abs(x.astype(float) - mid)[20:-20, 20:-20].mean()
    assert err(out) < err(linear)  # motion-aware models beat blending on a pure translation


def test_arbitrary_time_is_supported_and_deterministic():
    a, _, b = scene()
    x = external.interpolate('amt_s', a, b, .25)
    np.testing.assert_array_equal(x, external.interpolate('amt_s', a, b, .25))


def test_midpoint_only_model_rejects_other_times():
    if not external.available('rife_paper'):
        pytest.skip('paper RIFE export not present')
    a, _, b = scene(256, 448)
    with pytest.raises(ValueError):
        external.interpolate('rife_paper', a, b, .25)


def test_missing_shape_is_explicit():
    a, _, b = scene(200, 300)
    with pytest.raises(RuntimeError):
        external.interpolate('amt_s', a, b, .5)
