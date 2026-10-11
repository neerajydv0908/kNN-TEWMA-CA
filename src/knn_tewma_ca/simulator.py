"""Event-driven cluster simulator and the scheduler base class.

The event loop is shared by every scheduler. A scheduler only decides three
things: how long to allocate for the head of the queue, which waiting jobs to
backfill (and with what allocation), and what to learn when a job finishes.

Queue discipline (EASY-style)
-----------------------------
* Jobs wait in FCFS order. At each event the head of the queue starts if it
  fits on the free nodes.
* Otherwise the head gets a *reservation*: the earliest time enough nodes
  will be free for it. Other waiting jobs may jump ahead (backfill) only if
  they fit on the free nodes now and their allocation ends by the
  reservation.
* A job runs for ``min(allocated, actual)``. If the allocation is too short
  it is killed at the limit and counted as a failure.

Reservation guarantee
---------------------
Because a backfilled job is killed at its allocation, and its allocation ends
by the reservation, it always gives its nodes back in time. So backfilling
can never push the head job's reservation later. This holds for every
scheduler here by construction. ``Scheduler.run`` re-checks it after every
backfill round and raises ``ReservationGuaranteeError`` if it ever fails,
which would mean a bug in a scheduler, not a property of the workload.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence, Tuple

from .job import Job
from .metrics import Metrics, compute_metrics


class ReservationView(str, Enum):
    """What the scheduler knows about running jobs when it plans a reservation.

    ``TRUE_FINISH`` is the behaviour of the code behind the Round 2
    presentation: it reads each running job's real finish time, which a real
    scheduler cannot know. ``ALLOCATED_END`` only uses start time plus
    allocated runtime (the walltime limit), which is what a production
    scheduler such as SLURM plans with.
    """

    TRUE_FINISH = "true_finish"
    ALLOCATED_END = "allocated_end"


class ReservationGuaranteeError(RuntimeError):
    """Raised if a backfill round ever pushes the head job's reservation later."""


@dataclass
class Launch:
    """A decision to start ``job`` now with ``allocated`` runtime."""

    job: Job
    allocated: float


@dataclass
class SimulationResult:
    scheduler: str
    jobs: List[Job]
    metrics: Metrics
    backfill_rounds: int


#: Running-job heap entries: (true finish time, job id, job).
RunningHeap = List[Tuple[float, int, Job]]

_EPS = 1e-6


class Scheduler:
    """Base class. Subclasses implement the policy hooks below."""

    name = "scheduler"

    def __init__(
        self,
        total_nodes: int = 8,
        reservation_view: ReservationView = ReservationView.ALLOCATED_END,
    ) -> None:
        self.total_nodes = total_nodes
        self.reservation_view = ReservationView(reservation_view)

    # -- policy hooks -----------------------------------------------------
    def reset(self) -> None:
        """Clear learned state before a run."""

    def observe(self, job: Job) -> None:
        """Learn from a finished job."""

    def before_dispatch(
        self, now: float, waiting: Sequence[Job], free_nodes: int, running_jobs: List[Job]
    ) -> None:
        """Called once per event while the queue is non-empty, before dispatch."""

    def head_allocation(self, job: Job) -> float:
        """Runtime to allocate to the head of the queue when it starts."""
        raise NotImplementedError

    def backfill(
        self, now: float, waiting: Sequence[Job], free_nodes: int, reservation: float
    ) -> List[Launch]:
        """Pick jobs from ``waiting[1:]`` to start now, in launch order.

        Every launch must fit on ``free_nodes`` (cumulatively) and satisfy
        ``now + allocated <= reservation``.
        """
        raise NotImplementedError

    # -- simulator --------------------------------------------------------
    def reservation_time(self, now: float, free_nodes: int, needed: int, running: RunningHeap) -> float:
        """Earliest time at least ``needed`` nodes are free, from the scheduler's view."""
        if needed <= free_nodes:
            return now
        if self.reservation_view is ReservationView.TRUE_FINISH:
            ends = sorted(((finish, job) for finish, _, job in running), key=lambda x: x[0])
        else:
            ends = sorted(((job.allocated_end, job) for _, _, job in running), key=lambda x: x[0])
        accumulated = free_nodes
        for end, job in ends:
            accumulated += job.requested_nodes
            if accumulated >= needed:
                return end
        return math.inf

    @staticmethod
    def _launch(job: Job, now: float, allocated: float, running: RunningHeap) -> None:
        job.start_time = now
        job.allocated_runtime = allocated
        job.failed = allocated < job.actual_runtime
        finish = now + min(allocated, job.actual_runtime)
        heapq.heappush(running, (finish, job.job_id, job))

    def run(self, jobs: Sequence[Job]) -> SimulationResult:
        """Simulate ``jobs`` from an empty cluster. The input jobs are not modified."""
        self.reset()
        queue = sorted((j.fresh_copy() for j in jobs), key=lambda j: j.submit_time)
        waiting: List[Job] = []
        running: RunningHeap = []
        free_nodes = self.total_nodes
        idx = 0
        completed: List[Job] = []
        busy_node_time = 0.0
        backfill_rounds = 0

        while idx < len(queue) or waiting or running:
            next_submit = queue[idx].submit_time if idx < len(queue) else math.inf
            next_finish = running[0][0] if running else math.inf
            now = min(next_submit, next_finish)

            while running and running[0][0] <= now:
                finish, _, job = heapq.heappop(running)
                free_nodes += job.requested_nodes
                job.finish_time = finish
                completed.append(job)
                busy_node_time += min(job.actual_runtime, job.allocated_runtime) * job.requested_nodes
                self.observe(job)

            while idx < len(queue) and queue[idx].submit_time <= now:
                waiting.append(queue[idx])
                idx += 1

            if not waiting:
                continue

            self.before_dispatch(now, waiting, free_nodes, [job for _, _, job in running])

            head = waiting[0]
            if head.requested_nodes <= free_nodes:
                waiting.pop(0)
                self._launch(head, now, self.head_allocation(head), running)
                free_nodes -= head.requested_nodes
                continue

            reservation = self.reservation_time(now, free_nodes, head.requested_nodes, running)
            if head.promised_start == math.inf:
                head.promised_start = reservation

            launches = self.backfill(now, waiting, free_nodes, reservation)
            if not launches:
                continue

            for launch in launches:
                self._launch(launch.job, now, launch.allocated, running)
                free_nodes -= launch.job.requested_nodes
            started = {id(launch.job) for launch in launches}
            waiting[:] = [job for job in waiting if id(job) not in started]
            backfill_rounds += 1

            after = self.reservation_time(now, free_nodes, head.requested_nodes, running)
            if after > reservation + _EPS:
                raise ReservationGuaranteeError(
                    f"{self.name}: backfill at t={now:.3f} moved job {head.job_id}'s "
                    f"reservation from {reservation:.3f} to {after:.3f}"
                )

        metrics = compute_metrics(completed, busy_node_time, self.total_nodes)
        return SimulationResult(self.name, completed, metrics, backfill_rounds)
