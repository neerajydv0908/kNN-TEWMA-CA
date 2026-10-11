"""Runtime predictors and the queue-pressure signal.

``ClassModel`` keeps one runtime model per runtime class (short / medium /
long). It combines two estimators:

* **TEWMA** (trust-based EWMA): an exponentially weighted estimate of the
  relative prediction bias and its variance, turned into a confidence
  quantile of the runtime.
* **kNN refinement**: an inverse-distance weighted average of the
  actual/predicted ratio of the most similar past jobs. It replaces TEWMA's
  class-wide bias correction with a local one and keeps TEWMA's safety margin.

Killed jobs are censored observations: the scheduler only learns that the job
needed more than its allocation. ``ClassModel.observe`` reconstructs such a
job's runtime as its expected value under the class's error model, given that
it outlived its allocation, and learns from that.

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

# Censored observations ------------------------------------------------------
#: Lower limit on the relative error when reconstructing a killed job: a job
#: is assumed to run at most 20x its prediction.
MIN_RELATIVE_ERROR = -0.95
#: Floor on the error spread used for reconstruction, so the first killed jobs
#: of a class (before it has a variance estimate) are not pinned to their bound.
MIN_RECONSTRUCTION_STD = 0.1

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


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


def truncated_normal_moments(mu: float, sigma: float, lo: float, hi: float) -> Tuple[float, float]:
    """Mean and variance of N(mu, sigma^2) restricted to [lo, hi].

    Falls back to the tail asymptotics when the interval holds almost no
    probability mass (the mass then piles up at the edge nearest the mean).
    """
    if hi <= lo:
        return hi, 0.0
    a, b = (lo - mu) / sigma, (hi - mu) / sigma
    mass = _normal_cdf(b) - _normal_cdf(a)
    if mass > 1e-12:
        pa, pb = _normal_pdf(a), _normal_pdf(b)
        mean = mu + sigma * (pa - pb) / mass
        var = sigma * sigma * (1 + (a * pa - b * pb) / mass - ((pa - pb) / mass) ** 2)
        return min(hi, max(lo, mean)), max(0.0, var)
    if mu >= hi:
        shift = sigma * sigma / (mu - hi) if mu > hi else sigma
        return max(lo, hi - shift), shift * shift
    if mu <= lo:
        shift = sigma * sigma / (lo - mu) if mu < lo else sigma
        return min(hi, lo + shift), shift * shift
    return (lo + hi) / 2, 0.0  # a sliver of an interval around the mean


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

    def reconstruct(self, predicted: float, ran_for: float) -> Tuple[float, float]:
        """Expected relative error of a job killed after ``ran_for``, and its variance.

        The job's true runtime is longer than ``ran_for``, so its relative
        error ``(predicted - actual) / actual`` is below
        ``predicted / ran_for - 1``. Under the class's current error model
        N(bias, variance) this is a truncated normal; its mean and variance
        are what the model learns from (an online EM step).
        """
        hi = predicted / ran_for - 1
        sigma = max(math.sqrt(self.variance), MIN_RECONSTRUCTION_STD)
        return truncated_normal_moments(self.bias, sigma, MIN_RELATIVE_ERROR, hi)

    def observe(self, predicted: float, runtime: float, censored: bool = False) -> float:
        """Learn from a job that left the cluster.

        ``runtime`` is how long the job ran. If ``censored``, the job was
        killed and its true runtime is only known to be longer; the model
        reconstructs it first. Returns the runtime the model learned from.
        """
        if runtime <= 0:
            return runtime
        extra_variance = 0.0
        if censored:
            rel_error, extra_variance = self.reconstruct(predicted, runtime)
            runtime = predicted / (1 + rel_error)
        else:
            rel_error = (predicted - runtime) / runtime
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
        if censored:
            # Expected squared deviation also carries the reconstruction's own spread.
            sq_dev += extra_variance
        self.variance = (1 - self.alpha) * self.variance + self.alpha * sq_dev

        if predicted > 0:
            ratio = runtime / predicted
            self.max_observed_ratio = max(
                self.max_observed_ratio * RATIO_CAP_DECAY,
                ratio * RATIO_CAP_HEADROOM,
            )

        self.history.append((predicted, runtime))
        return runtime

    def safety_margin(self, confidence: float) -> float:
        """Multiplier that pads a point estimate to the given confidence level."""
        return max(0.5, 1.0 + inv_normal_cdf(confidence) * math.sqrt(self.variance))

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

    def knn_estimate(
        self,
        predicted: float,
        confidence: float = 0.80,
        k: int = DEFAULT_K,
        stack_on_bias_correction: bool = False,
    ) -> float:
        """kNN-refined runtime estimate.

        Finds the ``k`` past jobs whose predictions are closest in log space
        (so a 2s job and a 20s job are as far apart as a 20s and a 200s job)
        and takes an inverse-distance weighted average of their
        actual/predicted ratios. That local ratio replaces TEWMA's class-wide
        bias correction: the estimate is ``predicted x ratio x margin``, with
        the same confidence margin TEWMA uses. The result is clamped to
        ``[0.6, min(max_observed_ratio, 5)] x predicted``.

        ``stack_on_bias_correction=True`` reproduces the Round 2 version, which
        multiplied the ratio onto the already bias-corrected TEWMA estimate.
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

        if stack_on_bias_correction:
            corrected = self.tewma_estimate(predicted, confidence) * avg_ratio
        else:
            corrected = predicted * avg_ratio * self.safety_margin(confidence)
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
