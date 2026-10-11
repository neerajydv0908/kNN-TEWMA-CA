# Presentation numbers vs corrected methodology

* **Slides**: the Round 2 numbers (one seed, Round 2 methodology).
* **Corrected**: every methodology fix applied, mean over 30 seeds (0–29), with the 95% CI of the paired per-seed change.

## Headline claims

| Claim | Slides | Corrected, 30 seeds: mean [95% CI] | CA better in |
|:--|--:|--:|--:|
| E1 success rate, CA vs Baseline | +56.2% | +53.1% [+49.1, +57.2] | 30/30 |
| E1 utilization, CA vs Baseline | +6.0% | +0.9% [-0.1, +1.9] | 17/30 |
| E1 avg wait, CA vs Baseline | +13.8% | +45.8% [+36.1, +55.6] | 0/30 |
| E1 avg slowdown, CA vs Baseline | +2.0% | +29.8% [+20.9, +38.6] | 3/30 |
| E4 success rate, CA vs TEWMA | +26.4% | +24.1% [+22.4, +25.8] | 30/30 |
| E4 utilization, CA vs TEWMA | -1.9% | -3.8% [-4.7, -3.0] | 1/30 |
| E4 avg wait, CA vs TEWMA | +7.9% | -4.1% [-7.2, -1.1] | 21/30 |
| Reservation violations | 0 (reported as a result) | 0 for every scheduler, by construction | n/a |

## What each fix changes

The fixes applied one at a time, cumulatively, each over the same 30 seeds. The first row is the Round 2 code on 30 seeds instead of one, so the gap between it and the slides is the effect of the seed alone. Cells are means.

### E1 workload

B = Baseline, T = TEWMA, CA = kNN-TEWMA-CA.

| Step | Success: B | T | CA | Utilization: B | T | CA | Avg wait: B | T | CA | Alloc / needed: B | T | CA |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Round 2 code | 0.571 | 0.726 | 0.838 | 0.812 | 0.819 | 0.793 | 126.5 | 147.1 | 188.7 | 1.01 | 1.13 | 1.30 |
| + allocated-end reservations | 0.579 | 0.730 | 0.839 | 0.826 | 0.837 | 0.811 | 115.3 | 129.4 | 156.2 | 1.02 | 1.13 | 1.30 |
| + start every head job that fits | 0.581 | 0.731 | 0.838 | 0.858 | 0.875 | 0.862 | 93.9 | 105.7 | 127.5 | 1.02 | 1.13 | 1.30 |
| + censored learning | 0.529 | 0.671 | 0.782 | 0.858 | 0.870 | 0.856 | 91.4 | 99.7 | 111.8 | 1.01 | 1.07 | 1.13 |
| + kNN replaces the bias correction | 0.529 | 0.671 | 0.807 | 0.858 | 0.870 | 0.864 | 91.4 | 99.7 | 126.4 | 1.01 | 1.07 | 1.29 |

### E4 stress workload

B = Baseline, T = TEWMA, CA = kNN-TEWMA-CA.

| Step | Success: B | T | CA | Utilization: B | T | CA | Avg wait: B | T | CA | Alloc / needed: B | T | CA |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Round 2 code | 0.159 | 0.663 | 0.866 | 0.669 | 0.662 | 0.657 | 456.8 | 568.7 | 586.3 | 0.76 | 1.04 | 1.44 |
| + allocated-end reservations | 0.162 | 0.661 | 0.865 | 0.673 | 0.671 | 0.669 | 449.6 | 550.7 | 548.2 | 0.76 | 1.04 | 1.43 |
| + start every head job that fits | 0.224 | 0.670 | 0.865 | 0.905 | 0.910 | 0.877 | 276.1 | 332.3 | 333.4 | 0.77 | 1.04 | 1.43 |
| + censored learning | 0.130 | 0.578 | 0.828 | 0.907 | 0.913 | 0.879 | 269.5 | 327.3 | 324.4 | 0.75 | 0.96 | 1.26 |
| + kNN replaces the bias correction | 0.130 | 0.578 | 0.718 | 0.907 | 0.913 | 0.878 | 269.5 | 327.3 | 311.2 | 0.75 | 0.96 | 1.10 |

## Every metric

| Metric | Scheduler | Round 2 (slide seed) | Corrected, 30 seeds |
|:--|:--|--:|--:|
| E1 Makespan | Baseline | 2202.20 | 2250.0 ± 123.0 |
| E1 Makespan | TEWMA | 2197.99 | 2265.4 ± 126.1 |
| E1 Makespan | kNN-TEWMA-CA | 2211.59 | 2396.3 ± 154.6 |
| E1 Avg wait | Baseline | 70.66 | 91.4 ± 53.5 |
| E1 Avg wait | TEWMA | 74.37 | 99.7 ± 53.5 |
| E1 Avg wait | kNN-TEWMA-CA | 80.43 | 126.4 ± 59.9 |
| E1 Avg turnaround | Baseline | 89.67 | 112.1 ± 54.5 |
| E1 Avg turnaround | TEWMA | 94.08 | 120.9 ± 54.6 |
| E1 Avg turnaround | kNN-TEWMA-CA | 100.59 | 148.6 ± 61.1 |
| E1 Success rate | Baseline | 0.5400 | 0.529 ± 0.038 |
| E1 Success rate | TEWMA | 0.7233 | 0.671 ± 0.017 |
| E1 Success rate | kNN-TEWMA-CA | 0.8433 | 0.807 ± 0.021 |
| E1 Utilization | Baseline | 0.7600 | 0.858 ± 0.060 |
| E1 Utilization | TEWMA | 0.7913 | 0.870 ± 0.058 |
| E1 Utilization | kNN-TEWMA-CA | 0.8054 | 0.864 ± 0.047 |
| E1 Avg slowdown | Baseline | 11.46 | 13.31 ± 6.79 |
| E1 Avg slowdown | TEWMA | 12.18 | 15.01 ± 7.34 |
| E1 Avg slowdown | kNN-TEWMA-CA | 11.69 | 16.45 ± 6.60 |
| E1 Allocated / needed node-time | Baseline | 1.029 | 1.01 ± 0.05 |
| E1 Allocated / needed node-time | TEWMA | 1.194 | 1.07 ± 0.05 |
| E1 Allocated / needed node-time | kNN-TEWMA-CA | 1.370 | 1.29 ± 0.07 |
| E4 Makespan | Baseline | — | 898.4 ± 145.6 |
| E4 Makespan | TEWMA | 1528.65 | 1023.5 ± 154.1 |
| E4 Makespan | kNN-TEWMA-CA | 1632.48 | 1115.4 ± 168.9 |
| E4 Avg wait | Baseline | — | 269.5 ± 62.6 |
| E4 Avg wait | TEWMA | 545.61 | 327.3 ± 68.1 |
| E4 Avg wait | kNN-TEWMA-CA | 588.64 | 311.2 ± 60.7 |
| E4 Avg turnaround | Baseline | — | 286.9 ± 64.2 |
| E4 Avg turnaround | TEWMA | 566.39 | 347.2 ± 69.7 |
| E4 Avg turnaround | kNN-TEWMA-CA | 610.48 | 332.0 ± 62.4 |
| E4 Success rate | Baseline | — | 0.130 ± 0.035 |
| E4 Success rate | TEWMA | 0.7067 | 0.578 ± 0.025 |
| E4 Success rate | kNN-TEWMA-CA | 0.8933 | 0.718 ± 0.030 |
| E4 Utilization | Baseline | — | 0.907 ± 0.039 |
| E4 Utilization | TEWMA | 0.6711 | 0.913 ± 0.034 |
| E4 Utilization | kNN-TEWMA-CA | 0.6584 | 0.878 ± 0.029 |
| E4 Avg slowdown | Baseline | — | 39.66 ± 8.68 |
| E4 Avg slowdown | TEWMA | 93.45 | 50.18 ± 10.98 |
| E4 Avg slowdown | kNN-TEWMA-CA | 102.88 | 38.83 ± 7.93 |
| E4 Allocated / needed node-time | Baseline | — | 0.75 ± 0.07 |
| E4 Allocated / needed node-time | TEWMA | 1.070 | 0.96 ± 0.08 |
| E4 Allocated / needed node-time | kNN-TEWMA-CA | 1.463 | 1.10 ± 0.08 |

