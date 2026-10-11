import heapq
import math
from dataclasses import replace

import pytest

from knn_tewma_ca import (
    CORRECTED,
    DEFAULT_WORKLOAD,
    ROUND2,
    STRESS_WORKLOAD,
    EasyBaseline,
    Job,
    KnnTewmaCA,
    PaddedBaseline,
    ReservationView,
    TewmaScheduler,
    generate_jobs,
    make_scheduler,
)
from knn_tewma_ca.experiments import ABLATION_STEPS
from knn_tewma_ca.schedulers import SCHEDULER_NAMES

METHODOLOGIES = {key: m for key, _label, m in ABLATION_STEPS}
CASES = [
    (name, key, workload, seed)
    for name in SCHEDULER_NAMES
    for key in ("round2", "corrected")
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


@pytest.mark.parametrize("name, key, workload, seed", CASES)
def test_schedule_is_physically_valid(name, key, workload, seed):
    jobs = generate_jobs(workload, 8, seed)
    result = make_scheduler(name, 8, 0.8, METHODOLOGIES[key]).run(jobs)
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
    assert m.allocation_ratio == pytest.approx(
        sum(j.allocated_runtime * j.requested_nodes for j in done)
        / sum(j.actual_runtime * j.requested_nodes for j in done)
    )


def test_input_jobs_are_not_modified():
    jobs = generate_jobs(DEFAULT_WORKLOAD.with_jobs(80), 8, 1)
    snapshot = [j.fresh_copy() for j in jobs]
    KnnTewmaCA().run(jobs)
    assert jobs == snapshot


def test_runs_are_deterministic():
    jobs = generate_jobs(DEFAULT_WORKLOAD, 8, 11)
    assert KnnTewmaCA().run(jobs).metrics == KnnTewmaCA().run(jobs).metrics


@pytest.mark.parametrize("i", range(1, len(ABLATION_STEPS)))
def test_every_fix_changes_something(i):
    jobs = generate_jobs(STRESS_WORKLOAD, 8, 3)
    before = KnnTewmaCA(methodology=ABLATION_STEPS[i - 1][2]).run(jobs).metrics
    after = KnnTewmaCA(methodology=ABLATION_STEPS[i][2]).run(jobs).metrics
    assert before != after


class Recorder(KnnTewmaCA):
    """Records every observation and the true job it came from."""

    def reset(self):
        super().reset()
        self.seen = []

    def observe(self, obs):
        self.seen.append(obs)
        super().observe(obs)


def test_killed_jobs_only_reveal_a_lower_bound():
    jobs = generate_jobs(STRESS_WORKLOAD, 8, 0)
    sched = Recorder(methodology=CORRECTED)
    result = sched.run(jobs)
    by_id = {j.job_id: j for j in result.jobs}
    killed = [o for o in sched.seen if by_id[o.job_id].failed]
    assert killed, "the stress workload should kill some jobs"
    for o in sched.seen:
        job = by_id[o.job_id]
        assert o.censored == job.failed
        if job.failed:
            assert o.runtime == job.allocated_runtime < job.actual_runtime
        else:
            assert o.runtime == job.actual_runtime


def test_round2_leaks_true_runtimes():
    jobs = generate_jobs(STRESS_WORKLOAD, 8, 0)
    sched = Recorder(methodology=ROUND2)
    result = sched.run(jobs)
    by_id = {j.job_id: j for j in result.jobs}
    assert not any(o.censored for o in sched.seen)
    assert all(o.runtime == by_id[o.job_id].actual_runtime for o in sched.seen)


def _same_time_burst():
    # Three 2-node jobs arrive together on an empty 8-node cluster.
    return [Job(i, 0.0, 10.0, 10.0, 2) for i in range(3)]


def test_all_fitting_head_jobs_start_at_once():
    done = EasyBaseline(8, CORRECTED).run(_same_time_burst()).jobs
    assert [j.start_time for j in done] == [0.0, 0.0, 0.0]


def test_round2_starts_one_head_job_per_event():
    done = EasyBaseline(8, ROUND2).run(_same_time_burst()).jobs
    assert sorted(j.start_time for j in done) == [0.0, 10.0, 20.0]


def test_padded_baseline_pads_every_allocation():
    done = PaddedBaseline(8, CORRECTED, factor=1.5).run(_same_time_burst()).jobs
    assert all(j.allocated_runtime == pytest.approx(15.0) for j in done)
    assert all(not j.failed for j in done)


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
    peek = EasyBaseline(8, replace(CORRECTED, reservation_view=ReservationView.TRUE_FINISH))
    honest = EasyBaseline(8, CORRECTED)
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
