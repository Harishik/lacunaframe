"""Scene-clustered statistics for method comparisons (protocol v1, section 7).

The scene is the unit of analysis: seeds and frames of one scene share content
and are never treated as independent samples. Rows are first averaged within
each scene, then CIs come from bootstrapping scenes.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable, Mapping, Sequence
import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n_scenes: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PairedComparison:
    method: str
    baseline: str
    delta: Interval
    wilcoxon_p: float | None  # None when fewer than 2 nonzero scene differences
    wins: int  # scenes where method > baseline
    holm_p: float | None = None

    @property
    def significant(self) -> bool:
        """Claim rule: the paired CI excludes 0."""
        return self.delta.low > 0 or self.delta.high < 0

    def to_dict(self) -> dict:
        return dict(method=self.method, baseline=self.baseline, delta=self.delta.to_dict(),
                    wilcoxon_p=self.wilcoxon_p, holm_p=self.holm_p, wins=self.wins, significant=self.significant)


def scene_means(rows: Iterable[Mapping], value: str, scene: str = 'sequence') -> dict[str, float]:
    """Average a metric within each scene (over seeds, regimes already filtered by caller)."""
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        v = row[value]
        if v is None or not np.isfinite(float(v)):
            raise ValueError(f'Non-finite {value} for scene {row[scene]}; filter or cap before aggregating')
        groups[str(row[scene])].append(float(v))
    return {k: float(np.mean(v)) for k, v in sorted(groups.items())}


def bootstrap_mean(values: Sequence[float], n_boot: int = 10_000, level: float = .95, seed: int = 0) -> Interval:
    """Percentile bootstrap CI of the mean over independent units (scenes)."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError('Need a nonempty 1-D sequence of scene values')
    if len(x) == 1:
        return Interval(float(x[0]), float('nan'), float('nan'), 1)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return Interval(float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha)), len(x))


def paired(method: Mapping[str, float], baseline: Mapping[str, float], names: tuple[str, str] = ('method', 'baseline'),
           n_boot: int = 10_000, seed: int = 0) -> PairedComparison:
    """Paired scene-level difference method − baseline over shared scenes."""
    shared = sorted(set(method) & set(baseline))
    if not shared:
        raise ValueError('No shared scenes to pair')
    diff = np.array([method[s] - baseline[s] for s in shared])
    nonzero = diff[diff != 0]
    p = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else None
    return PairedComparison(names[0], names[1], bootstrap_mean(diff, n_boot=n_boot, seed=seed), p, int((diff > 0).sum()))


def holm(comparisons: Sequence[PairedComparison]) -> list[PairedComparison]:
    """Holm–Bonferroni adjusted p-values (comparisons with p=None are left as None)."""
    indexed = [(c.wilcoxon_p, i) for i, c in enumerate(comparisons) if c.wilcoxon_p is not None]
    indexed.sort()
    m, running, adjusted = len(indexed), 0.0, {}
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return [PairedComparison(c.method, c.baseline, c.delta, c.wilcoxon_p, c.wins, adjusted.get(i)) for i, c in enumerate(comparisons)]
