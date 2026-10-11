import math
import random

import pytest

from knn_tewma_ca.job import Job
from knn_tewma_ca.predictors import ClassModel, QueuePressure, inv_normal_cdf, truncated_normal_moments


@pytest.mark.parametrize(
    "p, z",
    [(0.5, 0.0), (0.8, 0.8416), (0.9, 1.2816), (0.95, 1.6449), (0.2, -0.8416)],
)
def test_inv_normal_cdf(p, z):
    assert inv_normal_cdf(p) == pytest.approx(z, abs=5e-4)


def test_cold_start_pads_by_25_percent():
    model = ClassModel()
    model.observe(10, 10)
    model.observe(10, 10)
    assert model.tewma_estimate(20) == pytest.approx(25)
    assert model.knn_estimate(20) == pytest.approx(25)
    assert model.trust_score == 0.5


def test_knn_falls_back_to_tewma_until_k_jobs_seen():
    model = ClassModel()
    for _ in range(7):
        model.observe(10, 12)
    assert model.knn_estimate(10, k=8) == model.tewma_estimate(10)


def test_perfect_predictions_are_trusted():
    model = ClassModel()
    for _ in range(50):
        model.observe(20, 20)
    assert model.bias == pytest.approx(0, abs=1e-12)
    assert model.variance == pytest.approx(0, abs=1e-12)
    assert model.trust_score == pytest.approx(1.0)
    assert model.tewma_estimate(20) == pytest.approx(20)
    assert model.knn_estimate(20) == pytest.approx(20)


def test_learns_systematic_under_prediction():
    model = ClassModel()
    for _ in range(300):
        model.observe(predicted=40, runtime=80)
    assert model.bias == pytest.approx(-0.5, abs=1e-6)
    # TEWMA removes the bias: 40 / (1 - 0.5) = 80.
    assert model.tewma_estimate(40) == pytest.approx(80, rel=1e-6)
    # kNN replaces the class-wide bias correction with the neighbours' ratio
    # (2x), so the bias is corrected once.
    assert model.knn_estimate(40) == pytest.approx(80, rel=1e-6)
    # The Round 2 version stacked the ratio on the bias-corrected estimate
    # (4x), and only the class cap (ratio 2 x 1.1 headroom) bounded it.
    assert model.knn_estimate(40, stack_on_bias_correction=True) == pytest.approx(40 * 2.2, rel=1e-6)
    assert model.trust_score < 0.6


def test_alpha_stays_in_range_and_tracks_volatility():
    rng = random.Random(0)
    calm, noisy = ClassModel(), ClassModel()
    for _ in range(40):
        actual = rng.uniform(10, 50)
        calm.observe(actual * (1 + rng.gauss(0, 0.02)), actual)
        noisy.observe(max(1.0, actual * (1 + rng.gauss(0, 1.0))), actual)
    assert 0.1 <= calm.alpha <= 0.6
    assert 0.1 <= noisy.alpha <= 0.6
    assert noisy.alpha > calm.alpha


def test_knn_estimate_is_clamped():
    rng = random.Random(1)
    for trial in range(200):
        model = ClassModel()
        for _ in range(rng.randint(8, 120)):
            actual = rng.uniform(1, 200)
            model.observe(max(1.0, actual * (1 + rng.gauss(0, 0.8))), actual)
        pred = rng.uniform(1, 200)
        est = model.knn_estimate(pred, confidence=rng.uniform(0.6, 0.95))
        assert 0.6 * pred - 1e-9 <= est <= 5.0 * pred + 1e-9
        assert 0.0 <= model.trust_score <= 1.0


def test_ratio_cap_decays_and_jumps():
    model = ClassModel()
    model.observe(10, 10)
    assert model.max_observed_ratio == pytest.approx(3.0 * 0.99)
    model.observe(10, 40)
    assert model.max_observed_ratio == pytest.approx(4.0 * 1.1)


def test_window_factor_shape():
    qp = QueuePressure()
    levels = [i / 100 for i in range(101)]
    factors = []
    for p in levels:
        qp.pressure_ema = p
        factors.append(qp.window_factor())
    assert factors[0] == 1.0 and factors[20] == 1.0
    assert factors[-1] == pytest.approx(0.70)
    assert all(b <= a for a, b in zip(factors, factors[1:]))
    assert all(0.7 <= f <= 1.0 for f in factors)


def test_trust_gate_range():
    qp = QueuePressure()
    for p in (0.0, 0.5, 1.0):
        qp.pressure_ema = p
        assert qp.trust_gate() == pytest.approx(0.2 * p)


def test_pressure_rises_with_backlog_and_decays_when_idle():
    qp = QueuePressure()
    running = [Job(0, 0.0, 10, 10, 4)]
    backlog = [Job(i, 0.0, 50, 50, 4) for i in range(1, 10)]
    for _ in range(20):
        qp.update(backlog, free_nodes=1, running_jobs=running)
    assert qp.pressure_ema == pytest.approx(1.0, abs=0.01)
    for _ in range(20):
        qp.update([], free_nodes=8, running_jobs=[])
    assert qp.pressure_ema < 0.01
    assert math.isfinite(qp.window_factor())


def _numeric_truncated_moments(mu, sigma, lo, hi, steps=200000):
    width = (hi - lo) / steps
    xs = [lo + (i + 0.5) * width for i in range(steps)]
    ws = [math.exp(-0.5 * ((x - mu) / sigma) ** 2) for x in xs]
    total = sum(ws)
    mean = sum(w * x for w, x in zip(ws, xs)) / total
    var = sum(w * (x - mean) ** 2 for w, x in zip(ws, xs)) / total
    return mean, var


@pytest.mark.parametrize(
    "mu, sigma, lo, hi",
    [(0.0, 0.3, -0.95, 0.0), (0.1, 0.2, -0.95, -0.2), (-0.3, 0.5, -0.95, -0.25), (0.0, 1.0, -0.95, 2.0)],
)
def test_truncated_normal_moments_match_numeric_integration(mu, sigma, lo, hi):
    mean, var = truncated_normal_moments(mu, sigma, lo, hi)
    num_mean, num_var = _numeric_truncated_moments(mu, sigma, lo, hi)
    assert mean == pytest.approx(num_mean, abs=1e-4)
    assert var == pytest.approx(num_var, abs=1e-4)


def test_truncated_normal_far_tail_stays_inside_the_interval():
    mean, var = truncated_normal_moments(2.0, 0.1, -0.95, -0.5)  # 25 sigma away
    assert -0.95 <= mean <= -0.5
    assert mean > -0.51  # mass piles up at the edge nearest the mean
    assert var >= 0.0


def test_censored_observation_is_reconstructed_beyond_its_bound():
    model = ClassModel()
    for _ in range(10):
        model.observe(20, 22)
    learned = model.observe(20, 25, censored=True)  # killed after 25
    assert learned > 25
    assert model.history[-1] == (20, learned)


def test_censoring_still_learns_that_jobs_run_long():
    # Every job really runs 1.5x its prediction and is killed at 1.2x. The
    # model never sees a true runtime, but it should still learn to allocate
    # more than the 1.2x it keeps getting killed at.
    model = ClassModel()
    for _ in range(60):
        model.observe(30, 36, censored=True)
    assert model.bias < 30 / 36 - 1
    assert model.tewma_estimate(30) > 36
    assert model.knn_estimate(30) > 36


def test_uncensored_path_is_unchanged():
    a, b = ClassModel(), ClassModel()
    rng = random.Random(3)
    for _ in range(50):
        actual = rng.uniform(1, 50)
        pred = max(1.0, actual * (1 + rng.gauss(0, 0.3)))
        a.observe(pred, actual)
        b.observe(pred, actual, censored=False)
    assert (a.bias, a.variance, a.alpha, list(a.history)) == (b.bias, b.variance, b.alpha, list(b.history))
