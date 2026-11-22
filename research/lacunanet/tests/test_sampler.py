import random
import numpy as np
import cv2
from research.lacunanet.data import GoproSource, StratifiedMotion, inner_motion


def _scene(v: float, n: int = 20, h: int = 360, w: int = 640):
    """Smooth texture translating v px/frame horizontally (known motion)."""
    rng = np.random.default_rng(int(v * 10))
    big = cv2.GaussianBlur(rng.integers(0, 255, (h + 40, w + 40 + int(v * n) + 40, 3)).astype(np.uint8), (0, 0), 3)
    big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    return np.stack([np.ascontiguousarray(big[20:20 + h, 20 + int(round(v * i)):20 + int(round(v * i)) + w]) for i in range(n)])


def test_stratified_sampler_matches_target_histogram():
    gopro = GoproSource.from_scenes([_scene(1.5), _scene(6.0)])
    bins = [0, 4, 16, 1e9]
    target = [0.5, 0.3, 0.2]
    sm = StratifiedMotion(gopro, None, target, bins, min_scale=0.72)
    assert sm.coverage == 1.0
    rng = random.Random(0)
    ms = [inner_motion(sm.sample(rng)[0]) for _ in range(150)]
    h, _ = np.histogram(ms, bins=bins)
    assert np.abs(h / h.sum() - np.array(target)).max() < 0.12, h / h.sum()
