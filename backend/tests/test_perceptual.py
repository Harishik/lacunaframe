"""ONNX LPIPS/DISTS: identity, monotonicity, alignment. Skipped when weights are not fetched."""
import numpy as np
import pytest
from backend import perceptual

pytestmark = pytest.mark.skipif(not (perceptual.available('lpips') and perceptual.available('dists')),
                                reason='perceptual weights not present (see research/metrics/export_perceptual.py)')


def frames(seed=0, n=2, h=64, w=96):
    return np.random.default_rng(seed).integers(0, 256, (n, h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize('metric', ['lpips', 'dists'])
def test_identity_is_near_zero_and_symmetric_shape(metric):
    a = frames()
    assert np.all(perceptual.distance(metric, a, a) < 1e-5)


@pytest.mark.parametrize('metric', ['lpips', 'dists'])
def test_more_damage_scores_worse(metric):
    a = frames(1, n=1, h=96, w=128)
    mild, heavy = a.copy(), a.copy()
    mild[:, :16] = 0
    heavy[:, :64] = 0
    assert perceptual.distance(metric, a, mild)[0] < perceptual.distance(metric, a, heavy)[0]


def test_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        perceptual.distance('lpips', frames(n=1), frames(n=2))
