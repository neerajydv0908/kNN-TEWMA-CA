"""Runtime predictors and the queue-pressure signal.

``ClassModel`` keeps one runtime model per runtime class (short / medium /
long). It combines two estimators:

* **TEWMA** (trust-based EWMA): an exponentially weighted estimate of the
  relative prediction bias and its variance, turned into a confidence
  quantile of the runtime.
* **kNN refinement**: an inverse-distance weighted average of the
  actual/predicted ratio of the most similar past jobs, applied on top of
  the TEWMA estimate.

``QueuePressure`` turns the state of the queue into a smoothed congestion
level in [0, 1] that the contention-aware scheduler uses to tighten its
backfill window and raise its confidence level.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Sequence, Tuple

from .job import Job

# TEWMA ------------------------------------------------------------------
INITIAL_ALPHA = 0.3
ALPHA_MIN = 0.1
ALPHA_MAX = 0.6
ALPHA_GAIN = 0.25
#: Roughly the error std of the noisiest job population. An error std this
#: large pushes alpha close to the top of its range.
ERROR_STD_SCALE = 0.5
MIN_ERRORS_FOR_ADAPTIVE_ALPHA = 4
RECENT_ERRORS = 20
WARMUP_OBSERVATIONS = 3
#: Padding used before a class has enough history to trust its statistics.
COLD_START_PADDING = 1.25

# kNN ----------------------------------------------------------------------
HISTORY_SIZE = 100
DEFAULT_K = 8
#: Starting value of the per-class ratio cap.
INITIAL_RATIO_CAP = 3.0
#: The cap never exceeds this multiple of the prediction.
MAX_RATIO_CAP = 5.0
#: Per-observation decay of the cap, so one outlier doesn't inflate it forever.
RATIO_CAP_DECAY = 0.99
#: Headroom added on top of a newly observed ratio.
RATIO_CAP_HEADROOM = 1.1
#: The kNN estimate is never below this fraction of the prediction.
KNN_FLOOR = 0.6


def inv_normal_cdf(p: float) -> float:
    """Approximate standard-normal quantile (Abramowitz & Stegun 26.2.23).

    Accurate to about 4.5e-4, which is plenty for picking a safety margin.
    """
    if p <= 0.5:
        sign, p = -1, 1 - p
    else:
        sign = 1
    t = math.sqrt(-2 * math.log(1 - p))
    c = (2.515517, 0.802853, 0.010328)
    d = (1.432788, 0.189269, 0.001308)
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    return sign * (t - num / den)


@dataclass
class ClassModel:
    """Runtime model for one runtime class."""

    alpha: float = INITIAL_ALPHA
    bias: float = 0.0
    variance: float = 0.0
    count: int = 0
    recent_errors: Deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_ERRORS))
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=HISTORY_SIZE))
    #: Data-driven cap on the actual/predicted ratio for this class. It starts
    #: at 3x, jumps up when a job runs longer than that, and otherwise decays
    #: slowly so a single outlier does not set the cap forever.
    max_observed_ratio: float = INITIAL_RATIO_CAP

    def observe(self, predicted: float, actual: float) -> None:
        """Learn from a finished job."""
        if actual <= 0:
            return
        rel_error = (predicted - actual) / actual
        self.recent_errors.append(rel_error)
        self.count += 1

        # Alpha follows the spread of recent errors: volatile predictions get
        # a faster-moving average, stable ones a smoother one.
        if len(self.recent_errors) >= MIN_ERRORS_FOR_ADAPTIVE_ALPHA:
            mu = sum(self.recent_errors) / len(self.recent_errors)
            recent_var = sum((e - mu) ** 2 for e in self.recent_errors) / len(self.recent_errors)
            self.alpha = max(
                ALPHA_MIN,
                min(ALPHA_MAX, ALPHA_MIN + (math.sqrt(recent_var) / ERROR_STD_SCALE) * ALPHA_GAIN),
            )

        self.bias = (1 - self.alpha) * self.bias + self.alpha * rel_error
        sq_dev = (rel_error - self.bias) ** 2
        self.variance = (1 - self.alpha) * self.variance + self.alpha * sq_dev

        if predicted > 0:
            ratio = actual / predicted
            self.max_observed_ratio = max(
                self.max_observed_ratio * RATIO_CAP_DECAY,
                ratio * RATIO_CAP_HEADROOM,
            )

        self.history.append((predicted, actual))

    def tewma_estimate(self, predicted: float, confidence: float = 0.80) -> float:
        """TEWMA runtime estimate at the given confidence level.

        Removes the learned bias from the prediction, then pads it by
        ``z(confidence)`` standard deviations of the relative error.
        """
        if self.count < WARMUP_OBSERVATIONS:
            return predicted * COLD_START_PADDING
        bias_corrected = predicted / max(0.1, 1 + self.bias)
        std = math.sqrt(self.variance)
        z = inv_normal_cdf(confidence)
        quantile_bound = bias_corrected * max(0.5, 1.0 + z * std)
        return max(predicted * 0.9, quantile_bound)

    def knn_estimate(self, predicted: float, confidence: float = 0.80, k: int = DEFAULT_K) -> float:
        """kNN-refined TEWMA estimate.

        Finds the ``k`` past jobs whose predictions are closest in log space
        (so a 2s job and a 20s job are as far apart as a 20s and a 200s job),
        takes an inverse-distance weighted average of their actual/predicted
        ratios, and scales the TEWMA estimate by it. The result is clamped to
        ``[0.6, min(max_observed_ratio, 5)] x predicted``.
        """
        if len(self.history) < max(k, WARMUP_OBSERVATIONS) or self.count < WARMUP_OBSERVATIONS:
            return self.tewma_estimate(predicted, confidence)

        dist_ratios = []
        for p_pred, p_act in self.history:
            if p_pred <= 0:
                continue
            dist = abs(math.log(max(predicted, 1.0) / max(p_pred, 1.0)))
            dist_ratios.append((dist, p_act / p_pred))

        if not dist_ratios:
            return self.tewma_estimate(predicted, confidence)

        # Stable sort: ties keep history order, oldest first.
        dist_ratios.sort(key=lambda x: x[0])

        weights, ratios = [], []
        for d, r in dist_ratios[: min(k, len(dist_ratios))]:
            weights.append(1.0 / (d + 0.001) if d > 0 else 100.0)
            ratios.append(r)

        total_w = sum(weights)
        avg_ratio = sum(w * r for w, r in zip(weights, ratios)) / total_w if total_w > 0 else 1.0

        corrected = self.tewma_estimate(predicted, confidence) * avg_ratio
        hard_cap = predicted * min(self.max_observed_ratio, MAX_RATIO_CAP)
        return max(predicted * KNN_FLOOR, min(corrected, hard_cap))

    @property
    def trust_score(self) -> float:
        """How much to trust this class's predictions, in [0, 1].

        Large bias or large variance lowers trust. Classes with too little
        history get a neutral 0.5.
        """
        if self.count < WARMUP_OBSERVATIONS:
            return 0.5
        return max(0.0, 1.0 - abs(self.bias) - 0.5 * math.sqrt(self.variance))


@dataclass
class QueuePressure:
    """Smoothed congestion level of the queue, in [0, 1].

    Raw pressure is the node-time the queue is asking for, divided by what
    the free nodes could deliver over one typical job length (the mean
    predicted runtime of the running jobs). It is smoothed with an EMA.
    """

    alpha: float = 0.25
    pressure_ema: float = 0.0
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    def update(self, waiting: Sequence[Job], free_nodes: int, running_jobs: List[Job]) -> float:
        if not waiting:
            raw = 0.0
        else:
            demanded = sum(j.predicted_runtime * j.requested_nodes for j in waiting)
            horizon = (
                sum(j.predicted_runtime for j in running_jobs) / len(running_jobs)
                if running_jobs
                else 1.0
            )
            available = max(1.0, free_nodes * horizon)
            raw = min(1.0, demanded / available)
        self.pressure_ema = (1 - self.alpha) * self.pressure_ema + self.alpha * raw
        self.history.append(self.pressure_ema)
        return self.pressure_ema

    def window_factor(self) -> float:
        """Fraction of the reservation window open to backfill.

        1.0 up to pressure 0.2, then ``1 - 0.3 * p**1.2``, reaching 0.7 at
        full pressure.
        """
        p = self.pressure_ema
        if p <= 0.2:
            return 1.0
        if p >= 1.0:
            return 0.70
        return 1.0 - 0.30 * (p ** 1.2)

    def trust_gate(self) -> float:
        """Minimum class trust score a job needs to be considered for backfill."""
        return max(0.0, 0.20 * self.pressure_ema)
