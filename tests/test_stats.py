import pytest

from knn_tewma_ca.stats import paired_comparison, summarize, t_critical_95


def test_t_critical_values():
    assert t_critical_95(29) == 2.045
    assert t_critical_95(1) == 12.706
    assert t_critical_95(60) == pytest.approx(2.000, abs=2e-3)
    assert t_critical_95(1000) == pytest.approx(1.962, abs=2e-3)


def test_summarize():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s.n == 4
    assert s.mean == 2.5
    assert s.std == pytest.approx(1.2909944)
    assert s.ci_low == pytest.approx(2.5 - 3.182 * 1.2909944 / 2)
    assert s.ci_high == pytest.approx(2.5 + 3.182 * 1.2909944 / 2)
    assert (s.minimum, s.maximum) == (1.0, 4.0)


def test_summarize_single_value():
    s = summarize([0.84])
    assert (s.mean, s.std, s.ci_low, s.ci_high) == (0.84, 0.0, 0.84, 0.84)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_paired_comparison_respects_direction():
    cand, ref = [0.8, 0.9, 0.7], [0.6, 0.6, 0.75]
    up = paired_comparison(cand, ref, higher_is_better=True)
    assert up.better == 2 and up.n == 3
    assert up.difference.mean == pytest.approx((0.2 + 0.3 - 0.05) / 3)
    down = paired_comparison(cand, ref, higher_is_better=False)
    assert down.better == 1


def test_paired_comparison_needs_equal_lengths():
    with pytest.raises(ValueError):
        paired_comparison([1.0], [1.0, 2.0], True)
