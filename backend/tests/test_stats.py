"""Properties of the scene-clustered statistics."""
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from backend.stats import bootstrap_mean, holm, paired, scene_means

finite = st.floats(-100, 100, allow_nan=False, allow_infinity=False)


@settings(max_examples=60, deadline=None)
@given(st.lists(finite, min_size=2, max_size=30))
def test_bootstrap_ci_brackets_mean_and_stays_in_range(values):
    ci = bootstrap_mean(values, n_boot=500)
    assert min(values) - 1e-9 <= ci.low <= ci.mean + 1e-9 <= ci.high + 2e-9 <= max(values) + 3e-9
    assert ci.n_scenes == len(values)


@settings(max_examples=40, deadline=None)
@given(st.lists(finite, min_size=2, max_size=20), finite)
def test_paired_difference_is_shift_equivariant(values, shift):
    base = {f's{i}': v for i, v in enumerate(values)}
    moved = {k: v + shift for k, v in base.items()}
    c = paired(moved, base, n_boot=300)
    assert c.delta.mean == pytest.approx(shift, abs=1e-6)


def test_scene_means_average_seeds_not_pool_them():
    rows = [dict(sequence='a', x=10.), dict(sequence='a', x=20.), dict(sequence='b', x=0.)]
    assert scene_means(rows, 'x') == {'a': 15., 'b': 0.}


def test_scene_means_reject_infinite_scores():
    with pytest.raises(ValueError):
        scene_means([dict(sequence='a', x=float('inf'))], 'x')


def test_consistent_large_effect_is_significant_and_wins_all():
    base = {f's{i}': float(i) for i in range(12)}
    better = {k: v + 2 + 0.01 * i for i, (k, v) in enumerate(base.items())}
    c = paired(better, base)
    assert c.significant and c.wins == 12 and c.wilcoxon_p < 0.001


def test_holm_adjustment_is_monotone_and_bounded():
    base = {f's{i}': 0. for i in range(10)}
    comps = [paired({k: d + 0.01 * j for j, k in enumerate(base)}, base, (f'm{d}', 'b')) for d in (0.0, 0.5, 2.0)]
    adjusted = holm(comps)
    for c in adjusted:
        if c.wilcoxon_p is not None:
            assert c.wilcoxon_p <= c.holm_p <= 1


def test_single_scene_has_no_interval():
    ci = bootstrap_mean([3.0])
    assert ci.mean == 3.0 and np.isnan(ci.low) and np.isnan(ci.high)
