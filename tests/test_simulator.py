import heapq
import math

import pytest

from knn_tewma_ca import (
    DEFAULT_WORKLOAD,
    STRESS_WORKLOAD,
    EasyBaseline,
    Job,
    KnnTewmaCA,
    ReservationView,
    TewmaScheduler,
    generate_jobs,
    make_scheduler,
)
from knn_tewma_ca.schedulers import SCHEDULER_NAMES

VIEWS = list(ReservationView)
CASES = [
    (name, view, workload, seed)
    for name in SCHEDULER_NAMES
    for view in VIEWS
    for workload, seed in ((DEFAULT_WORKLOAD, 0), (DEFAULT_WORKLOAD, 42), (STRESS_WORKLOAD, 7))
]


def _peak_nodes(jobs):
    # Finishes sort before starts at the same instant: a job ending at t
    # frees its nodes for a job starting at t.
    events = [(j.start_time, 1, j.requested_nodes) for j in jobs]
    events += [(j.finish_time, 0, -j.requested_nodes) for j in jobs]
    in_use = peak = 0
    for _, _, delta in sorted(events):
        in_use += delta
        peak = max(peak, in_use)
    return peak


@pytest.mark.parametrize("name, view, workload, seed", CASES)
def test_schedule_is_physically_valid(name, view, workload, seed):
    jobs = generate_jobs(workload, 8, seed)
    result = make_scheduler(name, 8, 0.8, view).run(jobs)
    done = result.jobs

    assert sorted(j.job_id for j in done) == [j.job_id for j in jobs]
    assert _peak_nodes(done) <= 8
    for j in done:
        assert j.start_time >= j.submit_time
        assert j.allocated_runtime > 0
        assert j.finish_time == pytest.approx(j.start_time + min(j.allocated_runtime, j.actual_runtime))
        assert j.failed == (j.allocated_runtime < j.actual_runtime)

    m = result.metrics
    assert 0.0 <= m.success_rate <= 1.0
    assert 0.0 < m.utilization <= 1.0
    assert m.makespan == max(j.finish_time for j in done)


def test_input_jobs_are_not_modified():
    jobs = generate_jobs(DEFAULT_WORKLOAD.with_jobs(80), 8, 1)
    snapshot = [j.fresh_copy() for j in jobs]
    KnnTewmaCA().run(jobs)
    assert jobs == snapshot


def test_runs_are_deterministic():
    jobs = generate_jobs(DEFAULT_WORKLOAD, 8, 11)
    assert KnnTewmaCA().run(jobs).metrics == KnnTewmaCA().run(jobs).metrics


def test_reservation_view_matters():
    jobs = generate_jobs(DEFAULT_WORKLOAD, 8, 42)
    true_finish = KnnTewmaCA(reservation_view=ReservationView.TRUE_FINISH).run(jobs).metrics
    allocated_end = KnnTewmaCA(reservation_view=ReservationView.ALLOCATED_END).run(jobs).metrics
    assert true_finish != allocated_end


def _running(*jobs):
    heap = []
    for j in jobs:
        heapq.heappush(heap, (j.start_time + min(j.allocated_runtime, j.actual_runtime), j.job_id, j))
    return heap


def _started(job_id, start, allocated, actual, nodes):
    j = Job(job_id, 0.0, actual, allocated, nodes)
    j.start_time, j.allocated_runtime = start, allocated
    return j


def test_allocated_end_view_ignores_true_finish_times():
    # Allocated 10, really finishes at 4. Only the true-finish view can know that.
    running = _running(_started(0, 0.0, allocated=10.0, actual=4.0, nodes=4))
    peek = EasyBaseline(8, ReservationView.TRUE_FINISH)
    honest = EasyBaseline(8, ReservationView.ALLOCATED_END)
    assert peek.reservation_time(1.0, 4, 8, running) == 4.0
    assert honest.reservation_time(1.0, 4, 8, running) == 10.0


def test_reservation_accumulates_nodes_in_end_order():
    running = _running(
        _started(0, 0.0, allocated=30.0, actual=30.0, nodes=2),
        _started(1, 0.0, allocated=10.0, actual=10.0, nodes=1),
        _started(2, 0.0, allocated=20.0, actual=20.0, nodes=3),
    )
    s = TewmaScheduler(8)
    assert s.reservation_time(0.0, 2, 2, running) == 0.0  # fits now
    assert s.reservation_time(0.0, 2, 3, running) == 10.0
    assert s.reservation_time(0.0, 2, 6, running) == 20.0
    assert s.reservation_time(0.0, 2, 8, running) == 30.0
    assert s.reservation_time(0.0, 2, 9, running) == math.inf
