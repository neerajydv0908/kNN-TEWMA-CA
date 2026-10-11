import math
import random

import pytest

from knn_tewma_ca.job import Job
from knn_tewma_ca.predictors import ClassModel, QueuePressure, inv_normal_cdf


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
        model.observe(predicted=40, actual=80)
    assert model.bias == pytest.approx(-0.5, abs=1e-6)
    # TEWMA removes the bias: 40 / (1 - 0.5) = 80.
    assert model.tewma_estimate(40) == pytest.approx(80, rel=1e-6)
    # kNN multiplies the already bias-corrected TEWMA estimate by the
    # neighbours' ratio (2x again), so the class cap (ratio 2 x 1.1 headroom)
    # is what bounds it.
    assert model.knn_estimate(40) == pytest.approx(40 * 2.2, rel=1e-6)
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
