"""Recoverability features: bracket logic, shapes, observation-only behaviour."""
import numpy as np
from research.lacunabench.recoverability import CHANNELS, MAP, PAIR_ONLY, BracketFeatures, brackets, frame_summary, summary_names


def moving_scene(n=12, speed=3, h=180, w=320):
    rng = np.random.default_rng(0)
    import cv2
    noise = rng.random((h, w + n * speed, 3)).astype(np.float32)
    base = cv2.GaussianBlur(noise, (0, 0), 3)
    base = ((base - base.min()) / (base.max() - base.min()) * 255).astype(np.uint8)
    return np.stack([base[:, i * speed:i * speed + w] for i in range(n)])


def test_brackets_find_outer_anchors_and_skip_unbracketed():
    m = np.zeros(12, bool)
    m[[0, 4, 5, 6, 11]] = True
    out = brackets(m)
    assert out == [(3, 7, 2, 8, [4, 5, 6])]


def test_feature_shapes_and_broadcast_scalars():
    frames = moving_scene()
    f = BracketFeatures(frames, 2, 6, 1, 7)
    maps = f.maps(3)
    assert maps.shape == (len(CHANNELS), MAP[1], MAP[0])
    assert np.allclose(maps[CHANNELS.index('dist_anchor')], 1) and np.allclose(maps[CHANNELS.index('log_gap')], 2)
    assert frame_summary(maps).shape == (2 * len(CHANNELS),) and len(summary_names()) == 2 * len(CHANNELS)
    assert frame_summary(maps, PAIR_ONLY).shape == (2 * len(PAIR_ONLY),)


def test_uniform_translation_has_low_acceleration_and_correct_speed():
    frames = moving_scene(speed=2)   # 320-wide frames = working width; 2 px/frame, 8 px across the gap
    f = BracketFeatures(frames, 3, 7, 1, 9)
    centre = (slice(10, 35), slice(20, 60))
    speed = f.maps(5)[CHANNELS.index('flow_mag')][centre].mean()
    accel = f.maps(5)[CHANNELS.index('accel')][centre].mean()
    assert abs(speed - 2) < .4
    assert accel < .2


def test_features_never_read_missing_frames():
    frames = moving_scene()
    m = np.zeros(len(frames), bool)
    m[4:7] = True
    left, right, l2, r2, idx = brackets(m)[0]
    poisoned = frames.copy()
    poisoned[m] = 255
    a = BracketFeatures(frames, left, right, l2, r2).maps(idx[1])
    b = BracketFeatures(poisoned, left, right, l2, r2).maps(idx[1])
    np.testing.assert_array_equal(a, b)
