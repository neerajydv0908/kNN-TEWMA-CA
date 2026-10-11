# Main results: corrected methodology, 30 seeds

Each cell is the mean ± sample standard deviation over 30 workloads (seeds 0–29). Every scheduler runs on the same workloads, so the comparisons are paired by seed. Brackets are 95% confidence intervals for the mean difference (Student t). "Better in" counts the seeds where kNN-TEWMA-CA is strictly better. "Allocated / needed node-time" is sum(allocated x nodes) / sum(actual x nodes): how much capacity a scheduler reserves relative to what jobs needed (neither direction is simply better).

## E1 · Baseline comparison (300 jobs, 8 nodes, p = 0.80)

| Metric | Baseline | TEWMA | kNN-TEWMA-CA |
|:--|--:|--:|--:|
| Makespan | 2250.0 ± 123.0 | 2265.4 ± 126.1 | 2396.3 ± 154.6 |
| Avg wait | 91.4 ± 53.5 | 99.7 ± 53.5 | 126.4 ± 59.9 |
| Avg turnaround | 112.1 ± 54.5 | 120.9 ± 54.6 | 148.6 ± 61.1 |
| Success rate | 0.529 ± 0.038 | 0.671 ± 0.017 | 0.807 ± 0.021 |
| Utilization | 0.858 ± 0.060 | 0.870 ± 0.058 | 0.864 ± 0.047 |
| Avg slowdown | 13.31 ± 6.79 | 15.01 ± 7.34 | 16.45 ± 6.60 |
| Allocated / needed node-time | 1.01 ± 0.05 | 1.07 ± 0.05 | 1.29 ± 0.07 |

### kNN-TEWMA-CA minus the other schedulers, same seed

| Metric | vs Baseline: mean diff [95% CI] | relative | better in | vs TEWMA: mean diff [95% CI] | relative | better in |
|:--|--:|--:|--:|--:|--:|--:|
| Makespan | +146.3 [+119.1, +173.4] | +6.5% | 0/30 | +130.9 [+106.6, +155.3] | +5.8% | 0/30 |
| Avg wait | +35.0 [+29.1, +40.9] | +45.8% | 0/30 | +26.7 [+22.5, +30.8] | +31.1% | 0/30 |
| Avg turnaround | +36.5 [+30.5, +42.5] | +36.3% | 0/30 | +27.7 [+23.5, +31.9] | +25.0% | 0/30 |
| Success rate | +0.278 [+0.264, +0.291] | +53.1% | 30/30 | +0.135 [+0.127, +0.144] | +20.2% | 30/30 |
| Utilization | +0.006 [-0.002, +0.014] | +0.9% | 17/30 | -0.006 [-0.013, +0.001] | -0.6% | 8/30 |
| Avg slowdown | +3.14 [+2.28, +4.00] | +29.8% | 3/30 | +1.44 [+0.76, +2.13] | +14.4% | 7/30 |
| Allocated / needed node-time | +0.28 [+0.26, +0.30] | +27.6% | — | +0.22 [+0.20, +0.23] | +20.4% | — |

## E2 · Confidence sweep (kNN-TEWMA-CA, 300 jobs)

| Confidence | Success rate | Utilization | Avg wait | Avg slowdown | Allocated / needed node-time |
|:--|--:|--:|--:|--:|--:|
| p70 | 0.732 ± 0.022 | 0.858 ± 0.048 | 116.1 ± 57.1 | 15.34 ± 6.52 | 1.15 ± 0.06 |
| p75 | 0.765 ± 0.022 | 0.862 ± 0.048 | 119.5 ± 57.6 | 15.59 ± 6.43 | 1.20 ± 0.06 |
| p80 | 0.807 ± 0.021 | 0.864 ± 0.047 | 126.4 ± 59.9 | 16.45 ± 6.60 | 1.29 ± 0.07 |
| p85 | 0.815 ± 0.019 | 0.864 ± 0.045 | 127.6 ± 59.5 | 16.66 ± 6.81 | 1.30 ± 0.07 |
| p90 | 0.819 ± 0.016 | 0.866 ± 0.044 | 127.8 ± 60.6 | 16.63 ± 6.81 | 1.30 ± 0.07 |

## E3 · Load sensitivity (p = 0.80)

| Jobs | Scheduler | Success rate | Utilization | Avg wait | Avg slowdown |
|:--|:--|--:|--:|--:|--:|
| 50 | Baseline | 0.508 ± 0.071 | 0.694 ± 0.117 | 24.3 ± 23.8 | 4.29 ± 3.01 |
| 50 | TEWMA | 0.681 ± 0.035 | 0.716 ± 0.115 | 27.8 ± 25.6 | 5.04 ± 3.58 |
| 50 | kNN-TEWMA-CA | 0.748 ± 0.046 | 0.709 ± 0.110 | 28.3 ± 25.8 | 4.70 ± 2.95 |
| 100 | Baseline | 0.514 ± 0.052 | 0.778 ± 0.095 | 46.8 ± 35.0 | 7.64 ± 4.89 |
| 100 | TEWMA | 0.675 ± 0.027 | 0.794 ± 0.094 | 51.9 ± 37.3 | 8.47 ± 5.34 |
| 100 | kNN-TEWMA-CA | 0.776 ± 0.031 | 0.790 ± 0.084 | 55.8 ± 37.3 | 8.42 ± 4.63 |
| 200 | Baseline | 0.521 ± 0.042 | 0.821 ± 0.081 | 70.7 ± 44.7 | 10.93 ± 5.88 |
| 200 | TEWMA | 0.674 ± 0.019 | 0.834 ± 0.082 | 76.6 ± 45.8 | 12.03 ± 6.28 |
| 200 | kNN-TEWMA-CA | 0.798 ± 0.027 | 0.833 ± 0.072 | 92.5 ± 48.7 | 13.01 ± 5.62 |
| 400 | Baseline | 0.534 ± 0.036 | 0.874 ± 0.053 | 104.6 ± 56.2 | 15.02 ± 7.18 |
| 400 | TEWMA | 0.673 ± 0.013 | 0.885 ± 0.053 | 114.7 ± 56.7 | 17.10 ± 7.77 |
| 400 | kNN-TEWMA-CA | 0.814 ± 0.020 | 0.876 ± 0.039 | 155.3 ± 69.2 | 19.42 ± 7.58 |

## E4 · High-contention stress (150 jobs, λ = 1.2, bias = −0.25, noise × 0.8)

| Metric | Baseline | TEWMA | kNN-TEWMA-CA |
|:--|--:|--:|--:|
| Makespan | 898.4 ± 145.6 | 1023.5 ± 154.1 | 1115.4 ± 168.9 |
| Avg wait | 269.5 ± 62.6 | 327.3 ± 68.1 | 311.2 ± 60.7 |
| Avg turnaround | 286.9 ± 64.2 | 347.2 ± 69.7 | 332.0 ± 62.4 |
| Success rate | 0.130 ± 0.035 | 0.578 ± 0.025 | 0.718 ± 0.030 |
| Utilization | 0.907 ± 0.039 | 0.913 ± 0.034 | 0.878 ± 0.029 |
| Avg slowdown | 39.66 ± 8.68 | 50.18 ± 10.98 | 38.83 ± 7.93 |
| Allocated / needed node-time | 0.75 ± 0.07 | 0.96 ± 0.08 | 1.10 ± 0.08 |

### kNN-TEWMA-CA minus the other schedulers, same seed

| Metric | vs TEWMA: mean diff [95% CI] | relative | better in | vs Baseline: mean diff [95% CI] | relative | better in |
|:--|--:|--:|--:|--:|--:|--:|
| Makespan | +91.9 [+81.3, +102.6] | +9.0% | 0/30 | +217.1 [+202.8, +231.3] | +24.4% | 0/30 |
| Avg wait | -16.1 [-26.6, -5.5] | -4.1% | 21/30 | +41.7 [+33.9, +49.6] | +17.0% | 2/30 |
| Avg turnaround | -15.1 [-25.7, -4.6] | -3.7% | 19/30 | +45.1 [+37.2, +53.1] | +17.1% | 2/30 |
| Success rate | +0.139 [+0.130, +0.148] | +24.1% | 30/30 | +0.587 [+0.576, +0.598] | +488.6% | 30/30 |
| Utilization | -0.035 [-0.043, -0.028] | -3.8% | 1/30 | -0.029 [-0.038, -0.021] | -3.2% | 3/30 |
| Avg slowdown | -11.35 [-13.98, -8.71] | -21.4% | 29/30 | -0.82 [-2.65, +1.00] | -0.8% | 18/30 |
| Allocated / needed node-time | +0.14 [+0.12, +0.15] | +14.6% | — | +0.35 [+0.33, +0.36] | +46.6% | — |

## E5 · What success costs

Each row is one setting, averaged over seeds. "Fixed padding" is the baseline with every allocation multiplied by a constant factor; TEWMA and kNN-TEWMA-CA are swept over their base confidence p. Comparing rows with similar allocated node-time shows how much success each approach gets for the capacity it reserves; comparing rows with similar success shows what that success costs in queue time and utilization.

### E1 workload (unbiased predictions)

| Scheduler | Setting | Allocated / needed node-time | Success rate | Utilization | Avg wait |
|:--|:--|--:|--:|--:|--:|
| Fixed padding | × 1 | 1.00 | 0.504 | 0.858 | 89.4 |
| Fixed padding | × 1.1 | 1.10 | 0.637 | 0.870 | 103.3 |
| Fixed padding | × 1.2 | 1.21 | 0.740 | 0.877 | 113.2 |
| Fixed padding | × 1.3 | 1.31 | 0.811 | 0.882 | 121.5 |
| Fixed padding | × 1.4 | 1.41 | 0.853 | 0.885 | 127.7 |
| Fixed padding | × 1.5 | 1.51 | 0.885 | 0.886 | 134.9 |
| Fixed padding | × 1.75 | 1.76 | 0.926 | 0.892 | 145.1 |
| Fixed padding | × 2 | 2.01 | 0.945 | 0.893 | 151.2 |
| TEWMA | p = 0.60 | 0.98 | 0.539 | 0.858 | 84.9 |
| TEWMA | p = 0.70 | 1.01 | 0.602 | 0.864 | 91.1 |
| TEWMA | p = 0.80 | 1.07 | 0.671 | 0.870 | 99.7 |
| TEWMA | p = 0.90 | 1.16 | 0.750 | 0.877 | 108.9 |
| TEWMA | p = 0.95 | 1.24 | 0.797 | 0.879 | 116.3 |
| TEWMA | p = 0.99 | 1.41 | 0.853 | 0.884 | 129.7 |
| kNN-TEWMA-CA | p = 0.55 | 1.03 | 0.630 | 0.851 | 102.9 |
| kNN-TEWMA-CA | p = 0.65 | 1.10 | 0.699 | 0.856 | 111.2 |
| kNN-TEWMA-CA | p = 0.75 | 1.20 | 0.765 | 0.862 | 119.5 |
| kNN-TEWMA-CA | p = 0.80 | 1.29 | 0.807 | 0.864 | 126.4 |
| kNN-TEWMA-CA | p = 0.90 | 1.30 | 0.819 | 0.866 | 127.8 |

### E4 workload (users under-predict by 25%)

| Scheduler | Setting | Allocated / needed node-time | Success rate | Utilization | Avg wait |
|:--|:--|--:|--:|--:|--:|
| Fixed padding | × 1 | 0.75 | 0.130 | 0.907 | 269.5 |
| Fixed padding | × 1.1 | 0.83 | 0.230 | 0.906 | 289.5 |
| Fixed padding | × 1.2 | 0.91 | 0.351 | 0.908 | 307.6 |
| Fixed padding | × 1.3 | 0.98 | 0.469 | 0.910 | 317.0 |
| Fixed padding | × 1.4 | 1.06 | 0.580 | 0.910 | 327.1 |
| Fixed padding | × 1.5 | 1.13 | 0.664 | 0.907 | 334.8 |
| Fixed padding | × 1.75 | 1.32 | 0.804 | 0.908 | 344.8 |
| Fixed padding | × 2 | 1.51 | 0.871 | 0.909 | 349.1 |
| TEWMA | p = 0.60 | 0.88 | 0.456 | 0.911 | 315.4 |
| TEWMA | p = 0.70 | 0.92 | 0.516 | 0.912 | 320.4 |
| TEWMA | p = 0.80 | 0.96 | 0.578 | 0.913 | 327.3 |
| TEWMA | p = 0.90 | 1.03 | 0.650 | 0.908 | 330.8 |
| TEWMA | p = 0.95 | 1.08 | 0.699 | 0.908 | 334.2 |
| TEWMA | p = 0.99 | 1.19 | 0.763 | 0.907 | 338.9 |
| kNN-TEWMA-CA | p = 0.55 | 0.93 | 0.536 | 0.878 | 295.1 |
| kNN-TEWMA-CA | p = 0.65 | 0.98 | 0.599 | 0.878 | 301.0 |
| kNN-TEWMA-CA | p = 0.75 | 1.05 | 0.673 | 0.879 | 307.0 |
| kNN-TEWMA-CA | p = 0.80 | 1.10 | 0.718 | 0.878 | 311.2 |
| kNN-TEWMA-CA | p = 0.90 | 1.11 | 0.718 | 0.877 | 311.4 |


## Reservation guarantee

Reservation violations are not reported as a result. Every backfilled job is allocated a runtime that ends by the head job's reservation and is killed at its allocation, so backfilling cannot delay a reservation under any of the three schedulers. The simulator re-checks this after every backfill round and raises `ReservationGuaranteeError` if it ever fails. It held in all 1830 simulations above, as it must.

