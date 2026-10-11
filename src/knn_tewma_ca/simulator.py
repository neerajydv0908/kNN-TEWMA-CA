"""Event-driven cluster simulator and the scheduler base class.

The event loop is shared by every scheduler. A scheduler only decides three
things: how long to allocate for the head of the queue, which waiting jobs to
backfill (and with what allocation), and what to learn when a job finishes.

Queue discipline (EASY-style)
-----------------------------
* Jobs wait in FCFS order. At each event, jobs at the head of the queue start
  while they fit on the free nodes.
* The first head job that does not fit gets a *reservation*: the earliest time
  enough nodes will be free for it. Other waiting jobs may jump ahead
  (backfill) only if they fit on the free nodes now and their allocation ends
  by the reservation.
* A job runs for ``min(allocated, actual)``. If the allocation is too short
  it is killed at the limit and counted as a failure.

What a scheduler can know
-------------------------
Schedulers never read a job's true runtime. When a job leaves the cluster
they receive an ``Observation``: how long it ran and whether it was killed.
For a killed job that is only a lower bound on its true runtime (a censored
observation). The ``Methodology`` controls this and the other differences
between the Round 2 code and the corrected version.

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


@dataclass(frozen=True)
class Methodology:
    """Everything that differs between the Round 2 code and the corrected version.

    Attributes:
        reservation_view: how running jobs' end times are estimated when
            planning a reservation.
        censored_learning: if True, a killed job only tells the scheduler that
            it needed more than its allocation. If False (Round 2), schedulers
            learn the killed job's true runtime.
        start_all_fitting_heads: if True, every job at the head of the queue
            that fits starts at the same event, then backfilling runs. If False
            (Round 2), at most one head job starts per event, and starting one
            also skips backfilling; everything else waits for the next arrival
            or completion.
        knn_stacks_on_bias_correction: if True (Round 2), the kNN ratio
            multiplies the TEWMA estimate, which has already removed the class
            bias, so the same bias is corrected twice. If False, the kNN ratio
            replaces the class-wide bias correction.
    """

    reservation_view: ReservationView = ReservationView.ALLOCATED_END
    censored_learning: bool = True
    start_all_fitting_heads: bool = True
    knn_stacks_on_bias_correction: bool = False


#: The code behind the Round 2 presentation.
ROUND2 = Methodology(
    reservation_view=ReservationView.TRUE_FINISH,
    censored_learning=False,
    start_all_fitting_heads=False,
    knn_stacks_on_bias_correction=True,
)

#: The corrected methodology (the default).
CORRECTED = Methodology()


@dataclass(frozen=True)
class Observation:
    """What a scheduler learns when a job leaves the cluster.

    ``runtime`` is how long the job ran. If ``censored`` is True the job was
    killed at its allocation, so its true runtime is only known to be longer
    than ``runtime``.
    """

    job_id: int
    predicted_runtime: float
    requested_nodes: int
    runtime: float
    censored: bool


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

    def __init__(self, total_nodes: int = 8, methodology: Methodology = CORRECTED) -> None:
        self.total_nodes = total_nodes
        self.methodology = methodology

    @property
    def reservation_view(self) -> ReservationView:
        return self.methodology.reservation_view

    # -- policy hooks -----------------------------------------------------
    def reset(self) -> None:
        """Clear learned state before a run."""

    def observe(self, obs: Observation) -> None:
        """Learn from a job that left the cluster."""

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

    def _observation(self, job: Job) -> Observation:
        if self.methodology.censored_learning:
            ran = min(job.actual_runtime, job.allocated_runtime)
            return Observation(job.job_id, job.predicted_runtime, job.requested_nodes, ran, job.failed)
        # Round 2: the scheduler is told the true runtime even of killed jobs.
        return Observation(job.job_id, job.predicted_runtime, job.requested_nodes, job.actual_runtime, False)

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
        drain = self.methodology.start_all_fitting_heads

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
                self.observe(self._observation(job))

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
                if not drain:
                    # Round 2: nothing else starts until the next event.
                    continue
                while waiting and waiting[0].requested_nodes <= free_nodes:
                    head = waiting.pop(0)
                    self._launch(head, now, self.head_allocation(head), running)
                    free_nodes -= head.requested_nodes
                if not waiting:
                    continue
                head = waiting[0]

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
