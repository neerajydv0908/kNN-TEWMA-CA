"""Small statistics helpers for multi-seed results (standard library only)."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

# Two-sided 95% Student-t critical values for 1..30 degrees of freedom.
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def t_critical_95(df: int) -> float:
    """Two-sided 95% t critical value. Exact table up to df=30, then a close approximation."""
    if df < 1:
        return math.nan
    if df <= len(_T975):
        return _T975[df - 1]
    # Asymptotic expansion around the normal quantile; within 0.001 for df > 30.
    z = 1.959964
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


@dataclass(frozen=True)
class Summary:
    """Mean, sample standard deviation and 95% confidence interval of the mean."""

    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float
    minimum: float
    maximum: float

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2


def summarize(values: Sequence[float]) -> Summary:
    values = list(values)
    if not values:
        raise ValueError("no values to summarize")
    n = len(values)
    mean = statistics.fmean(values)
    if n == 1:
        return Summary(1, mean, 0.0, mean, mean, mean, mean)
    std = statistics.stdev(values)
    half = t_critical_95(n - 1) * std / math.sqrt(n)
    return Summary(n, mean, std, mean - half, mean + half, min(values), max(values))


@dataclass(frozen=True)
class PairedComparison:
    """Per-seed comparison of a candidate against a reference on the same workloads."""

    difference: Summary  # candidate - reference
    relative: Summary  # (candidate - reference) / reference, as a fraction
    better: int  # seeds where the candidate is strictly better
    n: int


def paired_comparison(
    candidate: Sequence[float], reference: Sequence[float], higher_is_better: bool
) -> PairedComparison:
    if len(candidate) != len(reference):
        raise ValueError("paired samples must have the same length")
    diffs = [c - r for c, r in zip(candidate, reference)]
    rel = [(c - r) / r for c, r in zip(candidate, reference) if r != 0]
    better = sum(1 for d in diffs if (d > 0 if higher_is_better else d < 0))
    return PairedComparison(summarize(diffs), summarize(rel), better, len(diffs))
