"""The backfill rule guarantees reservations; these tests check that, and that the check has teeth."""

import pytest

from knn_tewma_ca import (
    DEFAULT_WORKLOAD,
    STRESS_WORKLOAD,
    EasyBaseline,
    ReservationGuaranteeError,
    ReservationView,
    generate_jobs,
    make_scheduler,
)
from knn_tewma_ca.schedulers import SCHEDULER_NAMES
from knn_tewma_ca.simulator import Launch


@pytest.mark.parametrize("view", list(ReservationView))
@pytest.mark.parametrize("name", SCHEDULER_NAMES)
@pytest.mark.parametrize("workload", [DEFAULT_WORKLOAD, STRESS_WORKLOAD], ids=["default", "stress"])
def test_no_reserved_job_starts_late(name, view, workload):
    for seed in range(4):
        result = make_scheduler(name, 8, 0.8, view).run(generate_jobs(workload, 8, seed))
        promised = [j for j in result.jobs if j.promised_start != float("inf")]
        assert promised, "expected some jobs to wait for a reservation"
        assert result.backfill_rounds > 0
        for j in promised:
            assert j.start_time <= j.promised_start + 1e-9


class GreedyBackfill(EasyBaseline):
    """Backfills anything that fits right now, ignoring the reservation."""

    name = "greedy"

    def backfill(self, now, waiting, free_nodes, reservation):
        launches = []
        for job in waiting[1:]:
            if job.requested_nodes <= free_nodes:
                launches.append(Launch(job, job.predicted_runtime * 3))
                free_nodes -= job.requested_nodes
        return launches


def test_breaking_the_window_rule_is_caught():
    with pytest.raises(ReservationGuaranteeError):
        GreedyBackfill(8).run(generate_jobs(DEFAULT_WORKLOAD, 8, 42))
