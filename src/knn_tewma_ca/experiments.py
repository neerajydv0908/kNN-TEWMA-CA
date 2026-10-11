"""The experiments.

From the Round 2 presentation:

* **E1 Baseline comparison**: Baseline vs TEWMA vs kNN-TEWMA-CA on 300 jobs.
* **E2 Confidence sweep**: kNN-TEWMA-CA at p in {0.70, 0.75, 0.80, 0.85, 0.90}.
* **E3 Load sensitivity**: 50, 100, 200 and 400 jobs.
* **E4 High-contention stress**: 150 jobs at 8x the arrival rate with users
  under-predicting by 25%.

Added in the rebuild:

* **E5 What success costs**: how much success each scheduler buys per unit
  of reserved node-time and per unit of queue time, against a baseline that
  just pads every prediction by a fixed factor. Run on both the E1 and the E4
  workload.
* **Ablation**: E1 and E4 under each methodology fix in turn, to show what
  each one changes.

Modes:

* ``REPRODUCE``: exactly what produced the presentation numbers. Round 2
  methodology and one seed per experiment (42, 99, 7).
* ``corrected(n)``: the corrected methodology over ``n`` seeds. This is the
  main version.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .metrics import Metrics
from .schedulers import EasyBaseline, KnnTewmaCA, PaddedBaseline, TewmaScheduler, make_scheduler
from .simulator import CORRECTED, ROUND2, Methodology, ReservationView
from .workload import DEFAULT_WORKLOAD, STRESS_WORKLOAD, generate_jobs

TOTAL_NODES = 8
DEFAULT_CONFIDENCE = 0.80
CONFIDENCE_LEVELS = (0.70, 0.75, 0.80, 0.85, 0.90)
LOAD_SIZES = (50, 100, 200, 400)
DEFAULT_SEED_COUNT = 30

# E5 sweeps.
PAD_FACTORS = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0)
TEWMA_LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
CA_LEVELS = (0.55, 0.65, 0.75, 0.80, 0.90)
FIXED_PADDING = "Fixed padding"

ALL_SCHEDULERS = (EasyBaseline.name, TewmaScheduler.name, KnnTewmaCA.name)
CA_ONLY = (KnnTewmaCA.name,)
TEWMA_AND_CA = (TewmaScheduler.name, KnnTewmaCA.name)

WORKLOADS = {"default": DEFAULT_WORKLOAD, "stress": STRESS_WORKLOAD}

EXPERIMENTS = {
    "E1": "Baseline comparison",
    "E2": "Confidence sweep",
    "E3": "Load sensitivity",
    "E4": "High-contention stress",
    "E5": "What success costs",
}

#: Cumulative methodology fixes, in the order the ablation applies them.
ABLATION_STEPS: Tuple[Tuple[str, str, Methodology], ...] = (
    ("round2", "Round 2 code", ROUND2),
    (
        "allocated_end",
        "+ allocated-end reservations",
        replace(ROUND2, reservation_view=ReservationView.ALLOCATED_END),
    ),
    (
        "all_heads",
        "+ start every head job that fits",
        replace(ROUND2, reservation_view=ReservationView.ALLOCATED_END, start_all_fitting_heads=True),
    ),
    (
        "censored",
        "+ censored learning",
        replace(
            ROUND2,
            reservation_view=ReservationView.ALLOCATED_END,
            start_all_fitting_heads=True,
            censored_learning=True,
        ),
    ),
    ("corrected", "+ kNN replaces the bias correction", CORRECTED),
)


@dataclass(frozen=True)
class Mode:
    key: str
    title: str
    methodology: Methodology
    e1_seeds: Tuple[int, ...]
    e3_seeds: Tuple[int, ...]
    e4_seeds: Tuple[int, ...]
    e3_schedulers: Tuple[str, ...]
    e4_schedulers: Tuple[str, ...]
    include_e5: bool

    @property
    def seed_count(self) -> int:
        return len(self.e1_seeds)


REPRODUCE = Mode(
    key="reproduce",
    title="Round 2 presentation (reproduced)",
    methodology=ROUND2,
    e1_seeds=(42,),
    e3_seeds=(99,),
    e4_seeds=(7,),
    e3_schedulers=CA_ONLY,
    e4_schedulers=TEWMA_AND_CA,
    include_e5=False,
)


def corrected(n_seeds: int = DEFAULT_SEED_COUNT, first_seed: int = 0) -> Mode:
    """Main methodology: every fix applied, ``n_seeds`` seeds per experiment.

    E3 and E4 also run the Baseline and TEWMA so every result has a reference
    point on the same workloads.
    """
    if n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")
    seeds = tuple(range(first_seed, first_seed + n_seeds))
    return Mode(
        key="corrected",
        title=f"Corrected methodology, {n_seeds} seeds",
        methodology=CORRECTED,
        e1_seeds=seeds,
        e3_seeds=seeds,
        e4_seeds=seeds,
        e3_schedulers=ALL_SCHEDULERS,
        e4_schedulers=ALL_SCHEDULERS,
        include_e5=True,
    )


@dataclass(frozen=True)
class Task:
    """One simulation to run. Plain data, so it can go to a worker process."""

    mode: str
    experiment: str
    scheduler: str
    seed: int
    workload: str
    methodology: Methodology
    confidence: float = DEFAULT_CONFIDENCE
    n_jobs: Optional[int] = None
    pad_factor: float = 1.0
    x: float = math.nan


@dataclass(frozen=True)
class Run:
    """Result of one simulation."""

    mode: str
    experiment: str
    scheduler: str
    seed: int
    workload: str
    #: Swept parameter: confidence (E2, E5), jobs (E3), padding factor (E5), else NaN.
    x: float
    metrics: Metrics


def simulate(task: Task) -> Run:
    config = WORKLOADS[task.workload]
    if task.n_jobs is not None:
        config = config.with_jobs(task.n_jobs)
    jobs = generate_jobs(config, TOTAL_NODES, task.seed)
    if task.scheduler == FIXED_PADDING:
        sched = PaddedBaseline(TOTAL_NODES, task.methodology, task.pad_factor)
    else:
        sched = make_scheduler(task.scheduler, TOTAL_NODES, task.confidence, task.methodology)
    metrics = sched.run(jobs).metrics
    return Run(task.mode, task.experiment, task.scheduler, task.seed, task.workload, task.x, metrics)


def default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def run_tasks(tasks: Sequence[Task], workers: int = 1) -> List[Run]:
    """Run tasks, in parallel when ``workers > 1``. Results keep task order."""
    if workers <= 1 or len(tasks) < 2:
        return [simulate(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(simulate, tasks, chunksize=max(1, len(tasks) // (workers * 8))))


# -- task builders -------------------------------------------------------------
def e1_tasks(mode: Mode) -> List[Task]:
    return [
        Task(mode.key, "E1", name, seed, "default", mode.methodology)
        for seed in mode.e1_seeds
        for name in ALL_SCHEDULERS
    ]


def e2_tasks(mode: Mode, levels: Sequence[float] = CONFIDENCE_LEVELS) -> List[Task]:
    return [
        Task(mode.key, "E2", KnnTewmaCA.name, seed, "default", mode.methodology, confidence=p, x=p)
        for seed in mode.e1_seeds
        for p in levels
    ]


def e3_tasks(mode: Mode, sizes: Sequence[int] = LOAD_SIZES) -> List[Task]:
    return [
        Task(mode.key, "E3", name, seed, "default", mode.methodology, n_jobs=n, x=float(n))
        for seed in mode.e3_seeds
        for n in sizes
        for name in mode.e3_schedulers
    ]


def e4_tasks(mode: Mode) -> List[Task]:
    return [
        Task(mode.key, "E4", name, seed, "stress", mode.methodology)
        for seed in mode.e4_seeds
        for name in mode.e4_schedulers
    ]


def e5_tasks(mode: Mode) -> List[Task]:
    tasks = []
    for workload in WORKLOADS:
        for seed in mode.e1_seeds:
            for f in PAD_FACTORS:
                tasks.append(Task(mode.key, "E5", FIXED_PADDING, seed, workload, mode.methodology,
                                  pad_factor=f, x=f))
            for p in TEWMA_LEVELS:
                tasks.append(Task(mode.key, "E5", TewmaScheduler.name, seed, workload, mode.methodology,
                                  confidence=p, x=p))
            for p in CA_LEVELS:
                tasks.append(Task(mode.key, "E5", KnnTewmaCA.name, seed, workload, mode.methodology,
                                  confidence=p, x=p))
    return tasks


BUILDERS = {"E1": e1_tasks, "E2": e2_tasks, "E3": e3_tasks, "E4": e4_tasks, "E5": e5_tasks}

Progress = Optional[Callable[[str], None]]


def run_mode(mode: Mode, workers: int = 1, progress: Progress = None) -> Dict[str, List[Run]]:
    """Run every experiment of ``mode``. Returns runs keyed by experiment."""
    keys = [k for k in BUILDERS if k != "E5" or mode.include_e5]
    if progress:
        progress(f"  {mode.title}: {', '.join(keys)}")
    tasks = [t for k in keys for t in BUILDERS[k](mode)]
    runs = run_tasks(tasks, workers)
    return {k: [r for r in runs if r.experiment == k] for k in keys}


def run_ablation(
    seeds: Sequence[int],
    workers: int = 1,
    final: Optional[Dict[str, List[Run]]] = None,
    progress: Progress = None,
) -> Dict[str, Dict[str, List[Run]]]:
    """E1 and E4 under each cumulative methodology step.

    ``final`` can supply the corrected-mode E1/E4 runs (same seeds) so the
    last step is not simulated twice.
    """
    tasks = []
    for key, label, methodology in ABLATION_STEPS:
        if key == "corrected" and final is not None:
            continue
        if progress:
            progress(f"  ablation: {label}")
        for seed in seeds:
            for name in ALL_SCHEDULERS:
                tasks.append(Task(key, "E1", name, seed, "default", methodology))
                tasks.append(Task(key, "E4", name, seed, "stress", methodology))
    runs = run_tasks(tasks, workers)
    out: Dict[str, Dict[str, List[Run]]] = {}
    for key, _label, _m in ABLATION_STEPS:
        if key == "corrected" and final is not None:
            out[key] = {"E1": list(final["E1"]), "E4": list(final["E4"])}
        else:
            out[key] = {e: [r for r in runs if r.mode == key and r.experiment == e] for e in ("E1", "E4")}
    return out


# -- selection helpers ------------------------------------------------------
def select(
    runs: Sequence[Run],
    scheduler: Optional[str] = None,
    x: Optional[float] = None,
    workload: Optional[str] = None,
) -> List[Run]:
    out = []
    for r in runs:
        if scheduler is not None and r.scheduler != scheduler:
            continue
        if x is not None and not math.isclose(r.x, x):
            continue
        if workload is not None and r.workload != workload:
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
