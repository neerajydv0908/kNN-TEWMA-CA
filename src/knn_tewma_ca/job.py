"""Job model and runtime classes."""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Runtime classes, keyed on the *predicted* runtime (the only size a
#: scheduler can see before a job runs).
RUNTIME_CLASSES = ("short", "medium", "long")
SHORT_LIMIT = 10.0
MEDIUM_LIMIT = 50.0


def runtime_class(predicted_runtime: float) -> str:
    """Bucket a job by its predicted runtime: short (<10), medium (<50) or long."""
    if predicted_runtime < SHORT_LIMIT:
        return "short"
    if predicted_runtime < MEDIUM_LIMIT:
        return "medium"
    return "long"


@dataclass
class Job:
    """A single batch job.

    ``actual_runtime`` is ground truth and only the simulator may read it.
    Schedulers see ``predicted_runtime`` (the user's estimate) and whatever
    they learn from jobs that have already finished.

    A job is given ``allocated_runtime`` when it starts. It runs for
    ``min(allocated_runtime, actual_runtime)``: if the allocation is too short
    the job is killed at its limit and counted as failed, just like a
    walltime kill on a real cluster.
    """

    job_id: int
    submit_time: float
    actual_runtime: float
    predicted_runtime: float
    requested_nodes: int
    start_time: float = -1.0
    finish_time: float = -1.0
    failed: bool = False
    allocated_runtime: float = 0.0
    #: First start time promised to this job while it was the blocked head of
    #: the queue (``inf`` if it never had to wait for a reservation).
    promised_start: float = math.inf

    @property
    def allocated_end(self) -> float:
        """Latest possible finish time: start plus the allocated runtime."""
        return self.start_time + self.allocated_runtime

    @property
    def wait_time(self) -> float:
        return self.start_time - self.submit_time

    @property
    def turnaround_time(self) -> float:
        return self.finish_time - self.submit_time

    def fresh_copy(self) -> "Job":
        """Copy of the job with all scheduling state cleared."""
        return Job(
            self.job_id,
            self.submit_time,
            self.actual_runtime,
            self.predicted_runtime,
            self.requested_nodes,
        )
