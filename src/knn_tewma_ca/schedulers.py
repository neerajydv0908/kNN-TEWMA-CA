"""The three schedulers compared in the experiments.

* ``EasyBaseline``: EASY-style backfilling. The head job gets exactly its
  predicted runtime; backfill candidates are padded by 1.0x / 1.2x / 1.5x
  depending on the recent average prediction error.
* ``TewmaScheduler``: every allocation comes from the per-class TEWMA
  confidence quantile. Backfill candidates are ranked by
  ``estimate x nodes x trust``.
* ``KnnTewmaCA``: kNN-refined TEWMA allocations plus contention awareness.
  Queue pressure raises the confidence level, shrinks the backfill window
  and gates out low-trust classes; candidates are ranked to prefer short,
  narrow, high-trust jobs.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Sequence

from .job import RUNTIME_CLASSES, Job, runtime_class
from .predictors import ClassModel, QueuePressure
from .simulator import Launch, ReservationView, Scheduler

#: Backfill candidates whose estimate is below this fraction of the user's
#: prediction are skipped, so a backfill decision never rests on the model
#: cutting a job's runtime by more than 15%.
MIN_ESTIMATE_FRACTION = 0.85


class EasyBaseline(Scheduler):
    """EASY-style backfilling with an error-based padding multiplier."""

    name = "Baseline"
    ERROR_WINDOW = 50

    def reset(self) -> None:
        self.recent_errors: Deque[float] = deque(maxlen=self.ERROR_WINDOW)

    def observe(self, job: Job) -> None:
        err = abs(job.predicted_runtime - job.actual_runtime) / max(job.actual_runtime, 1.0)
        self.recent_errors.append(err)

    def head_allocation(self, job: Job) -> float:
        return job.predicted_runtime

    def padding(self) -> float:
        avg_error = sum(self.recent_errors) / len(self.recent_errors) if self.recent_errors else 0.0
        if avg_error < 0.2:
            return 1.0
        return 1.2 if avg_error < 0.4 else 1.5

    def backfill(
        self, now: float, waiting: Sequence[Job], free_nodes: int, reservation: float
    ) -> List[Launch]:
        mult = self.padding()
        launches: List[Launch] = []
        for job in waiting[1:]:
            allocated = job.predicted_runtime * mult
            if job.requested_nodes <= free_nodes and now + allocated <= reservation:
                launches.append(Launch(job, allocated))
                free_nodes -= job.requested_nodes
        return launches


class TewmaScheduler(Scheduler):
    """Per-class TEWMA runtime correction with trust-weighted backfill."""

    name = "TEWMA"

    def __init__(
        self,
        total_nodes: int = 8,
        confidence: float = 0.80,
        reservation_view: ReservationView = ReservationView.ALLOCATED_END,
    ) -> None:
        super().__init__(total_nodes, reservation_view)
        self.confidence = confidence

    def reset(self) -> None:
        self.models: Dict[str, ClassModel] = {c: ClassModel() for c in RUNTIME_CLASSES}

    def model_for(self, job: Job) -> ClassModel:
        return self.models[runtime_class(job.predicted_runtime)]

    def observe(self, job: Job) -> None:
        self.model_for(job).observe(job.predicted_runtime, job.actual_runtime)

    def head_allocation(self, job: Job) -> float:
        return self.model_for(job).tewma_estimate(job.predicted_runtime, self.confidence)

    def backfill(
        self, now: float, waiting: Sequence[Job], free_nodes: int, reservation: float
    ) -> List[Launch]:
        candidates = []
        for position, job in enumerate(waiting[1:], start=1):
            model = self.model_for(job)
            est = model.tewma_estimate(job.predicted_runtime, self.confidence)
            if job.requested_nodes > free_nodes:
                continue
            if now + est > reservation:
                continue
            if est < job.predicted_runtime * MIN_ESTIMATE_FRACTION:
                continue
            score = est * job.requested_nodes * model.trust_score
            candidates.append((score, position, job, est))
        return _greedy_by_score(candidates, free_nodes)


class KnnTewmaCA(Scheduler):
    """kNN-TEWMA runtime estimates with contention-aware backfill control."""

    name = "kNN-TEWMA-CA"
    #: How far queue pressure can raise the confidence level.
    PRESSURE_CONFIDENCE_GAIN = 0.15
    MAX_CONFIDENCE = 0.95

    def __init__(
        self,
        total_nodes: int = 8,
        confidence: float = 0.80,
        reservation_view: ReservationView = ReservationView.ALLOCATED_END,
        k: int = 8,
    ) -> None:
        super().__init__(total_nodes, reservation_view)
        self.confidence = confidence
        self.k = k

    def reset(self) -> None:
        self.models: Dict[str, ClassModel] = {c: ClassModel() for c in RUNTIME_CLASSES}
        self.pressure = QueuePressure()
        self.current_confidence = self.confidence

    def model_for(self, job: Job) -> ClassModel:
        return self.models[runtime_class(job.predicted_runtime)]

    def observe(self, job: Job) -> None:
        self.model_for(job).observe(job.predicted_runtime, job.actual_runtime)

    def before_dispatch(
        self, now: float, waiting: Sequence[Job], free_nodes: int, running_jobs: List[Job]
    ) -> None:
        self.pressure.update(waiting, free_nodes, running_jobs)
        # More congestion -> more conservative allocations.
        self.current_confidence = min(
            self.MAX_CONFIDENCE,
            self.confidence + self.PRESSURE_CONFIDENCE_GAIN * self.pressure.pressure_ema,
        )

    def estimate(self, job: Job) -> float:
        return self.model_for(job).knn_estimate(job.predicted_runtime, self.current_confidence, self.k)

    def head_allocation(self, job: Job) -> float:
        return self.estimate(job)

    def backfill(
        self, now: float, waiting: Sequence[Job], free_nodes: int, reservation: float
    ) -> List[Launch]:
        # Under pressure only part of the reservation window is open to
        # backfill, which leaves slack for the jobs most likely to overrun.
        window = now + (reservation - now) * self.pressure.window_factor()
        gate = self.pressure.trust_gate()

        candidates = []
        for position, job in enumerate(waiting[1:], start=1):
            model = self.model_for(job)
            if model.trust_score < gate:
                continue
            est = self.estimate(job)
            if job.requested_nodes > free_nodes:
                continue
            if now + est > window:
                continue
            if est < job.predicted_runtime * MIN_ESTIMATE_FRACTION:
                continue
            # Prefer short, narrow, high-trust jobs: they fit gaps most easily
            # and are least likely to be killed. (TEWMA's score multiplies by
            # the node count, which favours wide jobs instead.)
            score = (
                (1.0 / (est + 1.0))
                * (1.0 / (job.requested_nodes + 0.5))
                * (model.trust_score ** 1.5)
            )
            candidates.append((score, position, job, est))
        return _greedy_by_score(candidates, free_nodes)


def _greedy_by_score(candidates: list, free_nodes: int) -> List[Launch]:
    """Launch candidates in descending score order while nodes remain.

    The sort is stable, so equal scores keep queue order.
    """
    candidates.sort(key=lambda c: -c[0])
    launches: List[Launch] = []
    for _score, _position, job, est in candidates:
        if job.requested_nodes <= free_nodes:
            launches.append(Launch(job, est))
            free_nodes -= job.requested_nodes
    return launches


SCHEDULERS = {
    EasyBaseline.name: EasyBaseline,
    TewmaScheduler.name: TewmaScheduler,
    KnnTewmaCA.name: KnnTewmaCA,
}
SCHEDULER_NAMES = tuple(SCHEDULERS)


def make_scheduler(
    name: str,
    total_nodes: int = 8,
    confidence: float = 0.80,
    reservation_view: ReservationView = ReservationView.ALLOCATED_END,
) -> Scheduler:
    """Build a scheduler by display name."""
    if name == EasyBaseline.name:
        return EasyBaseline(total_nodes, reservation_view)
    if name in SCHEDULERS:
        return SCHEDULERS[name](total_nodes, confidence, reservation_view)
    raise KeyError(f"unknown scheduler {name!r}; choose from {', '.join(SCHEDULERS)}")
