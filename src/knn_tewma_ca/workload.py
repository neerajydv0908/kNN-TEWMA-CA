"""Synthetic workload generator.

Jobs arrive as a Poisson process. Each job is drawn from one of three
populations (60% short, 30% medium, 10% long) and its predicted runtime is
the actual runtime times ``1 + noise``, where the noise is Gaussian with a
per-population spread. ``bias`` shifts that noise to model users who
systematically under- or over-estimate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from .job import Job

# (cumulative probability, runtime range, prediction-noise std)
_POPULATIONS = (
    (0.6, (1.0, 10.0), 0.2),
    (0.9, (10.0, 50.0), 0.4),
    (1.0, (50.0, 200.0), 0.6),
)


@dataclass(frozen=True)
class WorkloadConfig:
    """Parameters of a synthetic workload.

    Attributes:
        n_jobs: number of jobs to generate.
        arrival_rate: Poisson arrival rate (jobs per time unit).
        bias: mean of the relative prediction error. Negative values mean
            users under-predict.
        noise_scale: multiplier on each population's noise std.
        max_nodes_per_job: jobs request 1..max_nodes_per_job nodes uniformly.
    """

    n_jobs: int = 300
    arrival_rate: float = 0.15
    bias: float = 0.0
    noise_scale: float = 1.0
    max_nodes_per_job: int = 4

    def with_jobs(self, n_jobs: int) -> "WorkloadConfig":
        return WorkloadConfig(
            n_jobs, self.arrival_rate, self.bias, self.noise_scale, self.max_nodes_per_job
        )


#: Default workload used by E1, E2 and E3.
DEFAULT_WORKLOAD = WorkloadConfig()

#: E4: 8x the arrival rate, users under-predict by 25% on average.
STRESS_WORKLOAD = WorkloadConfig(n_jobs=150, arrival_rate=1.2, bias=-0.25, noise_scale=0.8)


def generate_jobs(config: WorkloadConfig, total_nodes: int, seed: int) -> List[Job]:
    """Generate a reproducible list of jobs for ``config`` and ``seed``.

    Uses its own ``random.Random`` instance, so it never touches the global
    random state. The draw order is fixed; changing it changes every
    workload.
    """
    rng = random.Random(seed)
    jobs: List[Job] = []
    current_time = 0.0
    for job_id in range(config.n_jobs):
        current_time += rng.expovariate(config.arrival_rate)
        r = rng.random()
        for cumulative, (low, high), noise_std in _POPULATIONS:
            if r < cumulative:
                break
        actual = rng.uniform(low, high)
        noise = rng.gauss(config.bias, noise_std * config.noise_scale)
        predicted = max(1.0, actual * (1 + noise))
        nodes = rng.randint(1, min(config.max_nodes_per_job, total_nodes))
        jobs.append(Job(job_id, current_time, actual, predicted, nodes))
    return jobs
