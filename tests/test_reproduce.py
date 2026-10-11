"""Reproduce mode must give exactly the numbers in the Round 2 slides and write-up."""

import pytest

from knn_tewma_ca import experiments as ex

SLIDES_E1 = {
    "Baseline": dict(makespan=2202.20, avg_wait=70.66, avg_turnaround=89.67,
                     success_rate=0.5400, utilization=0.7600, avg_slowdown=11.46),
    "TEWMA": dict(makespan=2197.99, avg_wait=74.37, avg_turnaround=94.08,
                  success_rate=0.7233, utilization=0.7913, avg_slowdown=12.18),
    "kNN-TEWMA-CA": dict(makespan=2211.59, avg_wait=80.43, avg_turnaround=100.59,
                         success_rate=0.8433, utilization=0.8054, avg_slowdown=11.69),
}

# confidence -> (success rate, utilization, avg slowdown)
SLIDES_E2 = {
    0.70: (0.8100, 0.7876, 10.74),
    0.75: (0.8267, 0.7968, 11.37),
    0.80: (0.8433, 0.8054, 11.69),
    0.85: (0.8633, 0.8070, 11.89),
    0.90: (0.8667, 0.8072, 11.89),
}

# jobs -> (success rate, avg wait, avg slowdown)
SLIDES_E3 = {
    50: (0.8000, 14.33, 3.22),
    100: (0.8400, 65.78, 9.53),
    200: (0.8500, 98.43, 15.28),
    400: (0.8425, 193.32, 26.95),
}

SLIDES_E4 = {
    "TEWMA": dict(makespan=1528.65, avg_wait=545.61, avg_turnaround=566.39,
                  success_rate=0.7067, utilization=0.6711, avg_slowdown=93.45),
    "kNN-TEWMA-CA": dict(makespan=1632.48, avg_wait=588.64, avg_turnaround=610.48,
                         success_rate=0.8933, utilization=0.6584, avg_slowdown=102.88),
}


@pytest.fixture(scope="module")
def reproduced():
    return ex.run_mode(ex.REPRODUCE)


def _rounded(runs, scheduler, x=None):
    selected = ex.select(runs, scheduler, x)
    assert len(selected) == 1
    return selected[0].metrics.rounded()


@pytest.mark.parametrize("scheduler", list(SLIDES_E1))
def test_e1_matches_slides(reproduced, scheduler):
    assert _rounded(reproduced["E1"], scheduler) == SLIDES_E1[scheduler]


@pytest.mark.parametrize("confidence", list(SLIDES_E2))
def test_e2_matches_slides(reproduced, confidence):
    m = _rounded(reproduced["E2"], "kNN-TEWMA-CA", confidence)
    assert (m["success_rate"], m["utilization"], m["avg_slowdown"]) == SLIDES_E2[confidence]


@pytest.mark.parametrize("n_jobs", list(SLIDES_E3))
def test_e3_matches_slides(reproduced, n_jobs):
    m = _rounded(reproduced["E3"], "kNN-TEWMA-CA", float(n_jobs))
    assert (m["success_rate"], m["avg_wait"], m["avg_slowdown"]) == SLIDES_E3[n_jobs]


@pytest.mark.parametrize("scheduler", list(SLIDES_E4))
def test_e4_matches_slides(reproduced, scheduler):
    assert _rounded(reproduced["E4"], scheduler) == SLIDES_E4[scheduler]


def test_headline_improvements(reproduced):
    base = _rounded(reproduced["E1"], "Baseline")
    ca = _rounded(reproduced["E1"], "kNN-TEWMA-CA")
    assert round((ca["success_rate"] / base["success_rate"] - 1) * 100, 1) == 56.2
    assert round((ca["utilization"] / base["utilization"] - 1) * 100, 1) == 6.0
    tewma_s = _rounded(reproduced["E4"], "TEWMA")
    ca_s = _rounded(reproduced["E4"], "kNN-TEWMA-CA")
    assert round((ca_s["success_rate"] / tewma_s["success_rate"] - 1) * 100, 1) == 26.4
