# Presentation numbers vs corrected methodology

* **Slides**: the Round 2 numbers (one seed, reservations planned from true finish times).
* **Same seed, allocated-end**: identical workloads, but reservations only use what a real scheduler knows (start time + allocated runtime). This isolates that fix.
* **30 seeds**: the main version. Mean ± standard deviation over seeds 0–29.

## Headline claims

| Claim | Slides | Same seed, allocated-end | 30 seeds: mean [95% CI] | CA better in |
|:--|--:|--:|--:|--:|
| E1 success rate, CA vs Baseline | +56.2% | +53.6% | +45.6% [+41.7, +49.5] | 30/30 |
| E1 utilization, CA vs Baseline | +6.0% | +5.7% | -1.8% [-2.8, -0.8] | 7/30 |
| E1 avg wait, CA vs Baseline | +13.8% | +9.4% | +46.5% [+32.9, +60.2] | 0/30 |
| E1 avg slowdown, CA vs Baseline | +2.0% | +0.3% | +31.9% [+20.6, +43.2] | 2/30 |
| E4 success rate, CA vs TEWMA | +26.4% | +27.6% | +31.1% [+28.8, +33.3] | 30/30 |
| E4 utilization, CA vs TEWMA | -1.9% | -2.0% | -0.2% [-1.8, +1.5] | 14/30 |
| E4 avg wait, CA vs TEWMA | +7.9% | +4.0% | -0.2% [-3.9, +3.6] | 15/30 |
| Reservation violations | 0 (reported as a result) | 0 | 0 for every scheduler, by construction | n/a |

## Every metric

| Metric | Scheduler | Slides | Same seed, allocated-end | 30 seeds |
|:--|:--|--:|--:|--:|
| E1 Makespan | Baseline | 2202.20 | 2202.20 | 2344.4 ± 163.3 |
| E1 Makespan | TEWMA | 2197.99 | 2197.97 | 2387.1 ± 163.9 |
| E1 Makespan | kNN-TEWMA-CA | 2211.59 | 2211.59 | 2529.4 ± 202.0 |
| E1 Avg wait | Baseline | 70.66 | 66.60 | 115.3 ± 66.4 |
| E1 Avg wait | TEWMA | 74.37 | 70.02 | 129.4 ± 67.7 |
| E1 Avg wait | kNN-TEWMA-CA | 80.43 | 72.84 | 156.2 ± 69.5 |
| E1 Avg turnaround | Baseline | 89.67 | 85.65 | 136.1 ± 67.4 |
| E1 Avg turnaround | TEWMA | 94.08 | 89.70 | 150.8 ± 68.8 |
| E1 Avg turnaround | kNN-TEWMA-CA | 100.59 | 92.98 | 178.2 ± 70.6 |
| E1 Success rate | Baseline | 0.5400 | 0.5533 | 0.579 ± 0.038 |
| E1 Success rate | TEWMA | 0.7233 | 0.7300 | 0.730 ± 0.021 |
| E1 Success rate | kNN-TEWMA-CA | 0.8433 | 0.8500 | 0.839 ± 0.019 |
| E1 Utilization | Baseline | 0.7600 | 0.7605 | 0.826 ± 0.049 |
| E1 Utilization | TEWMA | 0.7913 | 0.7889 | 0.837 ± 0.046 |
| E1 Utilization | kNN-TEWMA-CA | 0.8054 | 0.8040 | 0.811 ± 0.040 |
| E1 Avg slowdown | Baseline | 11.46 | 10.92 | 16.60 ± 8.64 |
| E1 Avg slowdown | TEWMA | 12.18 | 11.93 | 19.37 ± 9.27 |
| E1 Avg slowdown | kNN-TEWMA-CA | 11.69 | 10.95 | 20.43 ± 7.98 |
| E4 Makespan | Baseline | — | 1320.91 | 1213.8 ± 185.2 |
| E4 Makespan | TEWMA | 1528.65 | 1520.40 | 1425.3 ± 204.9 |
| E4 Makespan | kNN-TEWMA-CA | 1632.48 | 1625.86 | 1532.4 ± 211.7 |
| E4 Avg wait | Baseline | — | 443.63 | 449.6 ± 108.5 |
| E4 Avg wait | TEWMA | 545.61 | 534.66 | 550.7 ± 117.1 |
| E4 Avg wait | kNN-TEWMA-CA | 588.64 | 556.22 | 548.2 ± 125.9 |
| E4 Avg turnaround | Baseline | — | 461.63 | 467.1 ± 109.9 |
| E4 Avg turnaround | TEWMA | 566.39 | 555.38 | 571.0 ± 118.8 |
| E4 Avg turnaround | kNN-TEWMA-CA | 610.48 | 578.03 | 570.0 ± 127.5 |
| E4 Success rate | Baseline | — | 0.1467 | 0.162 ± 0.042 |
| E4 Success rate | TEWMA | 0.7067 | 0.7000 | 0.661 ± 0.032 |
| E4 Success rate | kNN-TEWMA-CA | 0.8933 | 0.8933 | 0.865 ± 0.018 |
| E4 Utilization | Baseline | — | 0.6707 | 0.673 ± 0.040 |
| E4 Utilization | TEWMA | 0.6711 | 0.6739 | 0.671 ± 0.039 |
| E4 Utilization | kNN-TEWMA-CA | 0.6584 | 0.6605 | 0.669 ± 0.035 |
| E4 Avg slowdown | Baseline | — | 68.78 | 68.78 ± 16.74 |
| E4 Avg slowdown | TEWMA | 93.45 | 89.61 | 87.57 ± 19.45 |
| E4 Avg slowdown | kNN-TEWMA-CA | 102.88 | 89.96 | 76.78 ± 20.66 |

