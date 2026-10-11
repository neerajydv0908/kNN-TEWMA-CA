"""Scheduling metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

from .job import Job

#: Metric name -> True if higher is better, False if lower is better, None if
#: neither (allocation ratio: too low kills jobs, too high wastes reservations).
HIGHER_IS_BETTER: Dict[str, Optional[bool]] = {
    "makespan": False,
    "avg_wait": False,
    "avg_turnaround": False,
    "success_rate": True,
    "utilization": True,
    "avg_slowdown": False,
    "allocation_ratio": None,
}
METRIC_NAMES = tuple(HIGHER_IS_BETTER)

#: The metrics reported in the Round 2 slides and write-up.
SLIDE_METRICS = METRIC_NAMES[:6]

METRIC_LABELS = {
    "makespan": "Makespan",
    "avg_wait": "Avg wait",
    "avg_turnaround": "Avg turnaround",
    "success_rate": "Success rate",
    "utilization": "Utilization",
    "avg_slowdown": "Avg slowdown",
    "allocation_ratio": "Allocated / needed node-time",
}

# Decimal places used in the Round 2 write-up and slides.
_DISPLAY_DIGITS = {
    "makespan": 2,
    "avg_wait": 2,
    "avg_turnaround": 2,
    "success_rate": 4,
    "utilization": 4,
    "avg_slowdown": 2,
    "allocation_ratio": 3,
}


@dataclass(frozen=True)
class Metrics:
    """Unrounded metrics for one simulation run.

    * ``makespan``: time the last job finishes.
    * ``avg_wait``: mean of start - submit.
    * ``avg_turnaround``: mean of finish - submit.
    * ``success_rate``: fraction of jobs whose allocation covered their
      actual runtime (i.e. not killed).
    * ``utilization``: node-time spent running jobs / (makespan x nodes).
      Time a job ran before being killed counts as busy.
    * ``avg_slowdown``: mean of turnaround / max(actual runtime, 1).
    * ``allocation_ratio``: allocated node-time / needed node-time, i.e.
      sum(allocated x nodes) / sum(actual x nodes). Above 1 the scheduler
      reserved more capacity than jobs needed; below 1 it killed them. Weighting
      by node-time means padding a short job counts for less than padding a
      long one. It is an evaluation metric only; schedulers never see it.
    """

    makespan: float
    avg_wait: float
    avg_turnaround: float
    success_rate: float
    utilization: float
    avg_slowdown: float
    allocation_ratio: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def rounded(self) -> Dict[str, float]:
        """Values rounded the way the Round 2 write-up reports them."""
        return {k: round(v, _DISPLAY_DIGITS[k]) for k, v in self.as_dict().items()}


def display_digits(metric: str) -> int:
    return _DISPLAY_DIGITS[metric]


def compute_metrics(completed: Sequence[Job], busy_node_time: float, total_nodes: int) -> Metrics:
    n = len(completed)
    if n == 0:
        raise ValueError("no completed jobs")
    makespan = max(j.finish_time for j in completed)
    avg_wait = sum(j.start_time - j.submit_time for j in completed) / n
    avg_turnaround = sum(j.finish_time - j.submit_time for j in completed) / n
    success_rate = sum(0 if j.failed else 1 for j in completed) / n
    utilization = busy_node_time / (makespan * total_nodes) if makespan > 0 else 0.0
    avg_slowdown = sum((j.finish_time - j.submit_time) / max(j.actual_runtime, 1.0) for j in completed) / n
    allocation_ratio = sum(j.allocated_runtime * j.requested_nodes for j in completed) / sum(
        j.actual_runtime * j.requested_nodes for j in completed
    )
    return Metrics(makespan, avg_wait, avg_turnaround, success_rate, utilization, avg_slowdown, allocation_ratio)
