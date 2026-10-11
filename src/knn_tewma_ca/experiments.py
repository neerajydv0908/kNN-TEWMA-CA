"""The four experiments from the Round 2 presentation.

* **E1 Baseline comparison**: Baseline vs TEWMA vs kNN-TEWMA-CA on 300 jobs.
* **E2 Confidence sweep**: kNN-TEWMA-CA at p in {0.70, 0.75, 0.80, 0.85, 0.90}.
* **E3 Load sensitivity**: 50, 100, 200 and 400 jobs.
* **E4 High-contention stress**: 150 jobs at 8x the arrival rate with users
  under-predicting by 25%.

Each experiment runs under a ``Mode``:

* ``REPRODUCE``: exactly what produced the presentation numbers. One seed per
  experiment (42, 99, 7) and reservations planned from true finish times.
* ``SAME_SEEDS_ALLOCATED_END``: the presentation seeds, but reservations
  planned from allocated end times. Isolates the effect of that fix.
* ``corrected(n)``: allocated-end reservations over ``n`` seeds per
  experiment. This is the main version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .metrics import Metrics
from .schedulers import EasyBaseline, KnnTewmaCA, TewmaScheduler, make_scheduler
from .simulator import ReservationView
from .workload import DEFAULT_WORKLOAD, STRESS_WORKLOAD, WorkloadConfig, generate_jobs

TOTAL_NODES = 8
DEFAULT_CONFIDENCE = 0.80
CONFIDENCE_LEVELS = (0.70, 0.75, 0.80, 0.85, 0.90)
LOAD_SIZES = (50, 100, 200, 400)
DEFAULT_SEED_COUNT = 30

ALL_SCHEDULERS = (EasyBaseline.name, TewmaScheduler.name, KnnTewmaCA.name)
CA_ONLY = (KnnTewmaCA.name,)
TEWMA_AND_CA = (TewmaScheduler.name, KnnTewmaCA.name)

EXPERIMENTS = {
    "E1": "Baseline comparison",
    "E2": "Confidence sweep",
    "E3": "Load sensitivity",
    "E4": "High-contention stress",
}


@dataclass(frozen=True)
class Mode:
    key: str
    title: str
    reservation_view: ReservationView
    e1_seeds: Tuple[int, ...]
    e3_seeds: Tuple[int, ...]
    e4_seeds: Tuple[int, ...]
    e3_schedulers: Tuple[str, ...]
    e4_schedulers: Tuple[str, ...]

    @property
    def seed_count(self) -> int:
        return len(self.e1_seeds)


REPRODUCE = Mode(
    key="reproduce",
    title="Round 2 presentation (reproduced)",
    reservation_view=ReservationView.TRUE_FINISH,
    e1_seeds=(42,),
    e3_seeds=(99,),
    e4_seeds=(7,),
    e3_schedulers=CA_ONLY,
    e4_schedulers=TEWMA_AND_CA,
)

SAME_SEEDS_ALLOCATED_END = Mode(
    key="same_seeds_allocated_end",
    title="Presentation seeds, allocated-end reservations",
    reservation_view=ReservationView.ALLOCATED_END,
    e1_seeds=(42,),
    e3_seeds=(99,),
    e4_seeds=(7,),
    e3_schedulers=ALL_SCHEDULERS,
    e4_schedulers=ALL_SCHEDULERS,
)


def corrected(n_seeds: int = DEFAULT_SEED_COUNT, first_seed: int = 0) -> Mode:
    """Main methodology: allocated-end reservations, ``n_seeds`` seeds per experiment.

    E3 and E4 also run the Baseline and TEWMA so every result has a reference
    point on the same workloads.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    seeds = tuple(range(first_seed, first_seed + n_seeds))
    return Mode(
        key="corrected",
        title=f"Allocated-end reservations, {n_seeds} seeds",
        reservation_view=ReservationView.ALLOCATED_END,
        e1_seeds=seeds,
        e3_seeds=seeds,
        e4_seeds=seeds,
        e3_schedulers=ALL_SCHEDULERS,
        e4_schedulers=ALL_SCHEDULERS,
    )


@dataclass(frozen=True)
class Run:
    """One simulation: an experiment cell for one scheduler and one seed."""

    mode: str
    experiment: str
    scheduler: str
    seed: int
    #: Swept parameter: confidence (E2), number of jobs (E3), NaN otherwise.
    x: float
    metrics: Metrics


Progress = Optional[Callable[[str], None]]


def _simulate(
    mode: Mode,
    experiment: str,
    scheduler: str,
    seed: int,
    workload: WorkloadConfig,
    confidence: float = DEFAULT_CONFIDENCE,
    x: float = math.nan,
) -> Run:
    jobs = generate_jobs(workload, TOTAL_NODES, seed)
    sched = make_scheduler(scheduler, TOTAL_NODES, confidence, mode.reservation_view)
    return Run(mode.key, experiment, scheduler, seed, x, sched.run(jobs).metrics)


def run_e1(mode: Mode) -> List[Run]:
    return [
        _simulate(mode, "E1", name, seed, DEFAULT_WORKLOAD)
        for seed in mode.e1_seeds
        for name in ALL_SCHEDULERS
    ]


def run_e2(mode: Mode, levels: Sequence[float] = CONFIDENCE_LEVELS) -> List[Run]:
    return [
        _simulate(mode, "E2", KnnTewmaCA.name, seed, DEFAULT_WORKLOAD, confidence=p, x=p)
        for seed in mode.e1_seeds
        for p in levels
    ]


def run_e3(mode: Mode, sizes: Sequence[int] = LOAD_SIZES) -> List[Run]:
    return [
        _simulate(mode, "E3", name, seed, DEFAULT_WORKLOAD.with_jobs(n), x=float(n))
        for seed in mode.e3_seeds
        for n in sizes
        for name in mode.e3_schedulers
    ]


def run_e4(mode: Mode) -> List[Run]:
    return [
        _simulate(mode, "E4", name, seed, STRESS_WORKLOAD)
        for seed in mode.e4_seeds
        for name in mode.e4_schedulers
    ]


RUNNERS = {"E1": run_e1, "E2": run_e2, "E3": run_e3, "E4": run_e4}


def run_mode(mode: Mode, progress: Progress = None) -> Dict[str, List[Run]]:
    """Run E1-E4 under ``mode``. Returns runs keyed by experiment."""
    results: Dict[str, List[Run]] = {}
    for key, runner in RUNNERS.items():
        if progress:
            progress(f"  {mode.title}: {key} {EXPERIMENTS[key].lower()}")
        results[key] = runner(mode)
    return results


# -- selection helpers ------------------------------------------------------
def select(runs: Sequence[Run], scheduler: Optional[str] = None, x: Optional[float] = None) -> List[Run]:
    out = []
    for r in runs:
        if scheduler is not None and r.scheduler != scheduler:
            continue
        if x is not None and not math.isclose(r.x, x):
            continue
        out.append(r)
    return sorted(out, key=lambda r: r.seed)


def values(runs: Sequence[Run], metric: str) -> List[float]:
    return [getattr(r.metrics, metric) for r in runs]


def schedulers_in(runs: Sequence[Run]) -> List[str]:
    present = {r.scheduler for r in runs}
    return [s for s in ALL_SCHEDULERS if s in present]


def xs_in(runs: Sequence[Run]) -> List[float]:
    return sorted({r.x for r in runs if not math.isnan(r.x)})
