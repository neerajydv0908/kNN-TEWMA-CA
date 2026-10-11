# kNN-TEWMA-CA

**A contention-aware adaptive scheduler for HPC job backfilling**

Hi, I'm Neeraj, a master's student in Computer Science at the University of Kansas. This is my research project for EECS 700 (Algorithms for HPC). It's a backfilling scheduler that learns how wrong users' runtime estimates tend to be and uses that to make safer backfill decisions, plus the event-driven cluster simulator I built to test it against an EASY-style baseline and a TEWMA-only scheduler.

This repo is a cleaned-up rebuild of the code behind my Round 2 presentation. It reproduces the presentation numbers exactly, but the main results now use a corrected methodology, and some of the headline numbers changed. Both are shown side by side below.

## Results at a glance

| Claim | Round 2 slides (one seed) | Now (30 seeds, mean [95% CI]) |
|:--|:--|:--|
| Success rate, kNN-TEWMA-CA vs baseline | 0.540 → 0.843 (+56.2%) | 0.579 → 0.839 (+45.6% [+41.7, +49.5]), better on 30/30 seeds |
| Utilization, kNN-TEWMA-CA vs baseline | 0.760 → 0.805 (+6.0%) | 0.826 → 0.811 (−1.8% [−2.8, −0.8]), lowest of the three schedulers |
| Avg wait, kNN-TEWMA-CA vs baseline | 70.7 → 80.4 (+13.8%) | 115.3 → 156.2 (+46.5% [+32.9, +60.2]) |
| Stress-test success, kNN-TEWMA-CA vs TEWMA | 0.707 → 0.893 (+26.4%) | 0.661 → 0.865 (+31.1% [+28.8, +33.3]), better on 30/30 seeds |
| Reservation violations | 0, reported as a result | 0 by construction, for all three schedulers |

Relative changes in the right-hand column are averaged per seed, so they don't always equal the ratio of the two means.

The short version: the reliability gain holds up across seeds, and under heavy contention it comes with no extra queue time compared with TEWMA. The utilization gain does not hold up; seed 42 was a lucky draw. And on the normal workload the extra safety costs more queue time than the single-seed run suggested. There's also an open issue in what the predictors learn from that affects the success numbers; see [What's still on my list](#whats-still-on-my-list).

![E1 success rate and utilization across 30 seeds](results/figures/e1_efficiency.png)

## The problem

On an HPC cluster, users submit jobs with a node count and a runtime estimate, and the scheduler decides what runs when. Most production schedulers use some form of **EASY backfilling**:

1. Jobs wait in first-come, first-served order.
2. If the job at the head of the queue doesn't fit, it gets a **reservation**: the earliest time enough nodes will be free.
3. Smaller jobs further back may **backfill** (jump ahead) if they fit on the idle nodes now and will finish before that reservation.

Backfilling only works if runtime estimates are good, and they usually aren't. If the scheduler trusts an estimate that's too low, the job is killed when it hits its time limit. If the estimate is too high, nodes sit idle and backfill opportunities are lost. So the question this project asks is:

> How can backfilling stay safe and efficient when runtime predictions are uncertain?

## How kNN-TEWMA-CA works

The name lists the three pieces: **kNN** runtime refinement, **TEWMA** (trust-based exponentially weighted moving average) runtime correction, and **C**ontention **A**wareness.

### 1. TEWMA runtime correction

Jobs are bucketed by predicted runtime $\hat p$ into short (< 10), medium (10–50) and long (≥ 50) classes, and each class learns from its own finished jobs. For each finished job the relative error is

$$e = \frac{\hat p - \text{actual}}{\text{actual}} \qquad (e > 0: \text{over-predicted},\ e < 0: \text{under-predicted})$$

Each class keeps an exponentially weighted bias $b$ and variance $v$ of $e$. The smoothing factor adapts to how volatile recent errors are: $\alpha = \text{clip}(0.1 + 0.25\,\sigma_{20} / 0.5,\ 0.1,\ 0.6)$, where $\sigma_{20}$ is the spread of the last 20 errors. Noisy classes adapt quickly and stable ones smooth more.

The runtime the scheduler allocates at confidence level $c$ removes the bias and adds a safety margin of $z_c$ standard deviations:

$$r_{\text{TEWMA}} = \max\!\Big(0.9\,\hat p,\ \ \frac{\hat p}{1 + b}\cdot \max\big(0.5,\ 1 + z_c\sqrt{v}\big)\Big)$$

Until a class has seen three jobs it falls back to $1.25\,\hat p$. Each class also gets a **trust score** $\max(0,\ 1 - |b| - 0.5\sqrt{v})$.

### 2. kNN refinement

TEWMA corrects a whole class at once. The kNN step looks at the jobs most similar to this one: from the class's last 100 finished jobs it takes the $k = 8$ with the closest predictions in log space, $d = |\log(\hat p / \hat p_i)|$, so a 2-to-20 gap counts the same as 20-to-200. It averages their actual/predicted ratios $\rho_i$ with weights $1/(d + 0.001)$ and scales the TEWMA estimate:

$$r_{\text{kNN}} = \text{clip}\big(r_{\text{TEWMA}} \cdot \bar\rho,\ \ 0.6\,\hat p,\ \ \min(\rho_{\max}, 5)\,\hat p\big)$$

$\rho_{\max}$ is a per-class cap: the largest observed ratio plus 10% headroom, decaying 1% per job so a single outlier doesn't set it forever.

### 3. Contention awareness

Queue pressure $P \in [0, 1]$ is an EMA of how much node-time the queue is asking for, relative to what the free nodes could deliver over one typical job length. Pressure changes three things:

* **Confidence goes up:** $c = \min(0.95,\ c_0 + 0.15P)$. A congested queue gets more conservative allocations.
* **The backfill window shrinks:** only the first $w(P)$ of the time until the reservation is open to backfill, with $w = 1$ for $P \le 0.2$ and $1 - 0.3P^{1.2}$ above that (0.7 at full pressure).
* **Low-trust classes are gated out:** a class needs trust ≥ $0.2P$ for its jobs to backfill.

A backfill candidate has to fit on the free nodes now, end inside the window, and have an estimate of at least $0.85\,\hat p$. Candidates are ranked by

$$\text{score} = \frac{1}{r + 1}\cdot\frac{1}{\text{nodes} + 0.5}\cdot \text{trust}^{1.5}$$

so short, narrow, well-predicted jobs go first, and they are launched greedily while nodes remain.

### The comparison schedulers

* **Baseline (EASY-style):** the head job gets exactly its predicted runtime. Backfill candidates are padded by 1.0×, 1.2× or 1.5× depending on the average prediction error of the last 50 jobs, and backfilled in queue order.
* **TEWMA:** piece 1 only. Every allocation is $r_{\text{TEWMA}}$, and candidates are ranked by estimate × nodes × trust.

## The simulator

* **Cluster:** 8 identical nodes. A job runs for `min(allocated, actual)` runtime. If its allocation is too short it is killed at the limit and counts as a failure, like a walltime kill.
* **Workload:** Poisson arrivals (λ = 0.15 jobs per time unit). 60% short jobs (actual runtime uniform on 1–10), 30% medium (10–50), 10% long (50–200), each asking for 1–4 nodes. Predicted runtime = max(1, actual × (1 + ε)) with ε ~ N(bias, σ), where σ is 0.2 / 0.4 / 0.6 for short / medium / long.
* **Load:** at λ = 0.15 the cluster receives about 1.16× the work it can process, so the queue keeps growing. That's deliberate: it keeps backfilling busy, and it's why waits grow with the number of jobs.
* **Metrics:** makespan, average wait (start − submit), average turnaround (finish − submit), success rate (jobs not killed), utilization (busy node-time / (makespan × nodes)), and average slowdown (turnaround / max(actual, 1)).

## Experiments

Every chart below shows the 30-seed results. In the strip charts each faint dot is one seed, the bar is the mean with its 95% confidence interval, and the ◇ is the value from my Round 2 slides. In the line charts the band is the 95% CI of the mean. The full tables are in [`results/corrected.md`](results/corrected.md).

### E1 · Baseline comparison

300 jobs, 8 nodes, base confidence p = 0.80. The success chart is at the top of this README; here are the time metrics.

![E1 wait, turnaround and slowdown](results/figures/e1_time.png)

| Metric | Baseline | TEWMA | kNN-TEWMA-CA |
|:--|--:|--:|--:|
| Success rate | 0.579 ± 0.038 | 0.730 ± 0.021 | **0.839 ± 0.019** |
| Utilization | 0.826 ± 0.049 | **0.837 ± 0.046** | 0.811 ± 0.040 |
| Avg wait | **115.3 ± 66.4** | 129.4 ± 67.7 | 156.2 ± 69.5 |
| Avg slowdown | **16.60 ± 8.64** | 19.37 ± 9.27 | 20.43 ± 7.98 |

The success ranking is rock solid: kNN-TEWMA-CA beats both other schedulers on all 30 seeds. Utilization is a different story. It's the lowest of the three, and it beat the baseline on only 7 of 30 seeds. The safety also costs queue time: about 41 time units more average wait than the baseline (95% CI 33.5–48.4), and it waits longer on every seed.

### E2 · Confidence sweep

kNN-TEWMA-CA at p ∈ {0.70, 0.75, 0.80, 0.85, 0.90}.

![E2 confidence sweep](results/figures/e2_confidence.png)

Success climbs from 0.785 at p = 0.70 to 0.839 at p = 0.80, then flattens (0.844, 0.845), while utilization stays flat at about 0.81. The plateau has a simple cause: queue pressure stays high in this overloaded workload, so the pressure-adjusted confidence spends most of its time at its 0.95 ceiling once p ≥ 0.85, and raising p further changes nothing. p = 0.80 is still a sensible default. It's where the sweep stops paying off.

### E3 · Load sensitivity

50, 100, 200 and 400 jobs at p = 0.80. In the slides this experiment only ran kNN-TEWMA-CA; here all three schedulers run on the same workloads so there's a reference point.

![E3 load sensitivity](results/figures/e3_load.png)

kNN-TEWMA-CA's success rate stays between 0.80 and 0.84 across sizes, but the baseline (0.54–0.59) and TEWMA (≈ 0.73) are just as flat. So the stability comes from how each scheduler sizes allocations, not from contention awareness specifically. Waits grow with job count for everyone, because the cluster is overloaded. kNN-TEWMA-CA's wait grows fastest (190.6 at 400 jobs vs 135.7 for the baseline).

### E4 · High-contention stress

150 jobs at 8× the arrival rate (λ = 1.2), with users under-predicting by 25% on average (bias −0.25, noise × 0.8).

![E4 stress test](results/figures/e4_stress.png)

| Metric | Baseline | TEWMA | kNN-TEWMA-CA | CA − TEWMA [95% CI] |
|:--|--:|--:|--:|--:|
| Success rate | 0.162 ± 0.042 | 0.661 ± 0.032 | **0.865 ± 0.018** | +0.204 [+0.192, +0.216] |
| Utilization | 0.673 ± 0.040 | 0.671 ± 0.039 | 0.669 ± 0.035 | −0.002 [−0.012, +0.008] |
| Avg wait | **449.6 ± 108.5** | 550.7 ± 117.1 | 548.2 ± 125.9 | −2.5 [−20.3, +15.3] |
| Avg slowdown | **68.78 ± 16.74** | 87.57 ± 19.45 | 76.78 ± 20.66 | −10.79 [−15.90, −5.68] |

This is where kNN-TEWMA-CA looks best. It completes 20 percentage points more jobs than TEWMA (better on all 30 seeds) at the same utilization and the same average wait, with lower slowdown. The +7.9% wait penalty in my slides turns out to be specific to that one seed. The baseline collapses to 16% success because it mostly allocates exactly what the under-predicting users asked for.

## What changed since Round 2

The rebuild fixes three methodology problems in the presentation code. The corrected version is the default; `knn-tewma-ca reproduce` still runs the original methodology.

**1. Reservations only use information a real scheduler has.** The original code planned each reservation by walking through the running jobs in order of their *true* finish times. A real scheduler can't know those. It only knows when each job started and how much time it was allocated, and that's what production backfill planners such as SLURM's work from. The simulator now plans with `start + allocated runtime` (`ReservationView.ALLOCATED_END`). The old behaviour is kept as `ReservationView.TRUE_FINISH` so reproduce mode stays exact. On the presentation seed this fix alone changes little (kNN-TEWMA-CA success 0.8433 → 0.8500, utilization 0.8054 → 0.8040). Waits drop for all three schedulers, because planning with allocated end times puts reservations later and opens a wider backfill window.

**2. Zero reservation violations is a guarantee, not a measured result.** In the slides "0 reservation violations" was a headline result for kNN-TEWMA-CA. It can't come out any other way. Every backfilled job gets an allocation that ends by the head job's reservation, and it's killed if it runs past that allocation, so it always hands its nodes back in time. Backfilling therefore can't push a reservation later, under any of the three schedulers. The simulator re-checks this after every backfill round and raises `ReservationGuaranteeError` if it ever fails. The tests check that no reserved job ever starts after the time it was promised, and that a deliberately broken scheduler that ignores the window is caught (`tests/test_reservation_guarantee.py`).

**3. Thirty seeds instead of one.** Every number in the presentation came from a single random workload (seed 42 for E1/E2, 99 for E3, 7 for E4). Each experiment now runs on 30 workloads (seeds 0–29), with all schedulers on the same workloads. The tables report mean ± standard deviation, the comparisons are paired by seed with 95% confidence intervals, and I count how many seeds each scheduler wins. This is what showed that the utilization gain was a property of seed 42.

The full side-by-side comparison, including the "same seed, allocated-end reservations" column that isolates fix 1, is in [`results/comparison.md`](results/comparison.md).

## What's still on my list

* **The predictors learn from killed jobs' true runtimes.** When a job is killed at its allocation, the simulator still feeds its real runtime to the TEWMA and kNN models. A real scheduler only learns that the job needed *more* than it was given, so those should be treated as censored observations. The success-rate results depend heavily on this, so it's the next thing I'm fixing.
* **One queue-head start per event.** The event loop starts at most one job from the head of the queue per event. When several jobs could start at the same moment, the rest wait for the next arrival or completion. This inflates waits and lowers utilization for all three schedulers.
* **kNN-TEWMA-CA allocates generously.** The kNN ratio is applied on top of a TEWMA estimate that has already removed the class bias, so systematic under-prediction gets corrected twice until the class cap kicks in. Part of the success gain is simply larger allocations, and a baseline that pads every allocation by a fixed factor would be a fairer comparison.
* **Scope:** synthetic workloads instead of real traces, a small 8-node cluster, nodes as the only resource, no per-user fairness, and several hand-picked parameters.

Further out, I'd like to validate on real traces (ANL Intrepid, LLNL logs), add node count as a second kNN distance dimension, try it as a SLURM backfill plugin, replace the hand-tuned pressure curves with a learned (reinforcement learning) policy, and extend it to mixed CPU/GPU nodes.

## Running it

Requires Python 3.9+ and matplotlib.

```bash
git clone https://github.com/neerajydv0908/kNN-TEWMA-CA.git
cd kNN-TEWMA-CA
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

knn-tewma-ca              # everything: reproduce + corrected, reports and charts (~1-2 min)
knn-tewma-ca reproduce    # just the Round 2 numbers (a few seconds)
knn-tewma-ca corrected --seeds 50 --out results-50
pytest                    # 82 tests (~30 s)
```

Without installing, `PYTHONPATH=src python3 -m knn_tewma_ca` runs the same thing from the repo root.

| Output | What's in it |
|:--|:--|
| `results/reproduce.md` | The Round 2 tables, reproduced exactly |
| `results/corrected.md` | Main results: 30 seeds, mean ± std, paired comparisons |
| `results/comparison.md` | Slides vs same-seed-with-fix-1 vs 30 seeds, metric by metric |
| `results/runs.csv` | Every simulation run (mode, experiment, scheduler, seed, metrics) |
| `results/figures/` | Charts for the corrected results, plus `reproduce_*.png` redraws of the presentation charts |

Using the scheduler directly:

```python
from knn_tewma_ca import DEFAULT_WORKLOAD, KnnTewmaCA, generate_jobs

jobs = generate_jobs(DEFAULT_WORKLOAD, total_nodes=8, seed=0)
result = KnnTewmaCA(total_nodes=8, confidence=0.80).run(jobs)
print(result.metrics)
```

## Repository layout

```
src/knn_tewma_ca/
  job.py           job model and runtime classes
  workload.py      synthetic workload generator (seeded, no global random state)
  predictors.py    per-class TEWMA + kNN model, queue pressure
  simulator.py     event loop, reservations, the reservation guarantee check
  schedulers.py    Baseline, TEWMA and kNN-TEWMA-CA
  metrics.py       scheduling metrics
  experiments.py   E1-E4 and the reproduce / corrected modes
  stats.py         means, confidence intervals, paired comparisons
  report.py        markdown and CSV reports
  plotting.py      charts
  cli.py           the knn-tewma-ca command
tests/             unit tests, including exact reproduction of the slides
results/           generated reports and charts
```

## License

[MIT](LICENSE) © 2026 Neeraj Yadav
