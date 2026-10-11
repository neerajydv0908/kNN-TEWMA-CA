# Main results: allocated-end reservations, 30 seeds

Each cell is the mean ± sample standard deviation over 30 workloads (seeds 0–29). Every scheduler runs on the same workloads, so the comparisons below are paired by seed. Brackets are 95% confidence intervals for the mean difference (Student t). "Better in" counts the seeds where kNN-TEWMA-CA is strictly better.

## E1 · Baseline comparison (300 jobs, 8 nodes, p = 0.80)

| Metric | Baseline | TEWMA | kNN-TEWMA-CA |
|:--|--:|--:|--:|
| Makespan | 2344.4 ± 163.3 | 2387.1 ± 163.9 | 2529.4 ± 202.0 |
| Avg wait | 115.3 ± 66.4 | 129.4 ± 67.7 | 156.2 ± 69.5 |
| Avg turnaround | 136.1 ± 67.4 | 150.8 ± 68.8 | 178.2 ± 70.6 |
| Success rate | 0.579 ± 0.038 | 0.730 ± 0.021 | 0.839 ± 0.019 |
| Utilization | 0.826 ± 0.049 | 0.837 ± 0.046 | 0.811 ± 0.040 |
| Avg slowdown | 16.60 ± 8.64 | 19.37 ± 9.27 | 20.43 ± 7.98 |

### kNN-TEWMA-CA minus the other schedulers, same seed

| Metric | vs Baseline: mean diff [95% CI] | relative | better in | vs TEWMA: mean diff [95% CI] | relative | better in |
|:--|--:|--:|--:|--:|--:|--:|
| Makespan | +185.0 [+152.8, +217.1] | +7.9% | 0/30 | +142.3 [+114.6, +170.0] | +5.9% | 0/30 |
| Avg wait | +41.0 [+33.5, +48.4] | +46.5% | 0/30 | +26.8 [+20.7, +33.0] | +26.3% | 1/30 |
| Avg turnaround | +42.1 [+34.6, +49.7] | +37.3% | 0/30 | +27.4 [+21.2, +33.6] | +21.5% | 1/30 |
| Success rate | +0.260 [+0.244, +0.277] | +45.6% | 30/30 | +0.110 [+0.100, +0.120] | +15.1% | 30/30 |
| Utilization | -0.015 [-0.024, -0.007] | -1.8% | 7/30 | -0.026 [-0.033, -0.020] | -3.1% | 4/30 |
| Avg slowdown | +3.83 [+2.62, +5.03] | +31.9% | 2/30 | +1.06 [-0.14, +2.26] | +10.4% | 11/30 |

## E2 · Confidence sweep (kNN-TEWMA-CA, 300 jobs)

| Confidence | Success rate | Utilization | Avg wait | Avg slowdown |
|:--|--:|--:|--:|--:|
| p70 | 0.785 ± 0.019 | 0.811 ± 0.041 | 142.7 ± 66.2 | 18.96 ± 7.66 |
| p75 | 0.809 ± 0.019 | 0.812 ± 0.040 | 148.2 ± 64.8 | 19.59 ± 7.56 |
| p80 | 0.839 ± 0.019 | 0.811 ± 0.040 | 156.2 ± 69.5 | 20.43 ± 7.98 |
| p85 | 0.844 ± 0.016 | 0.811 ± 0.039 | 156.9 ± 69.1 | 20.48 ± 7.93 |
| p90 | 0.845 ± 0.015 | 0.811 ± 0.039 | 156.9 ± 69.0 | 20.46 ± 7.93 |

## E3 · Load sensitivity (p = 0.80)

| Jobs | Scheduler | Success rate | Utilization | Avg wait | Avg slowdown |
|:--|:--|--:|--:|--:|--:|
| 50 | Baseline | 0.541 ± 0.070 | 0.682 ± 0.107 | 28.2 ± 26.7 | 4.84 ± 3.34 |
| 50 | TEWMA | 0.735 ± 0.040 | 0.703 ± 0.104 | 32.9 ± 30.2 | 5.63 ± 3.96 |
| 50 | kNN-TEWMA-CA | 0.799 ± 0.041 | 0.693 ± 0.101 | 34.0 ± 28.9 | 5.50 ± 3.27 |
| 100 | Baseline | 0.551 ± 0.060 | 0.751 ± 0.082 | 56.7 ± 44.2 | 9.04 ± 6.10 |
| 100 | TEWMA | 0.732 ± 0.029 | 0.768 ± 0.076 | 62.8 ± 46.1 | 10.40 ± 6.94 |
| 100 | kNN-TEWMA-CA | 0.821 ± 0.025 | 0.759 ± 0.067 | 68.9 ± 47.4 | 10.53 ± 6.35 |
| 200 | Baseline | 0.569 ± 0.046 | 0.791 ± 0.071 | 87.0 ± 54.4 | 13.13 ± 7.14 |
| 200 | TEWMA | 0.732 ± 0.025 | 0.805 ± 0.071 | 98.2 ± 58.1 | 15.32 ± 7.91 |
| 200 | kNN-TEWMA-CA | 0.837 ± 0.021 | 0.790 ± 0.061 | 113.6 ± 58.8 | 15.90 ± 6.90 |
| 400 | Baseline | 0.586 ± 0.036 | 0.840 ± 0.048 | 135.7 ± 70.1 | 19.26 ± 9.16 |
| 400 | TEWMA | 0.732 ± 0.018 | 0.853 ± 0.047 | 153.3 ± 71.2 | 22.73 ± 9.60 |
| 400 | kNN-TEWMA-CA | 0.841 ± 0.015 | 0.828 ± 0.039 | 190.6 ± 77.6 | 24.08 ± 8.81 |

## E4 · High-contention stress (150 jobs, λ = 1.2, bias = −0.25, noise × 0.8)

| Metric | Baseline | TEWMA | kNN-TEWMA-CA |
|:--|--:|--:|--:|
| Makespan | 1213.8 ± 185.2 | 1425.3 ± 204.9 | 1532.4 ± 211.7 |
| Avg wait | 449.6 ± 108.5 | 550.7 ± 117.1 | 548.2 ± 125.9 |
| Avg turnaround | 467.1 ± 109.9 | 571.0 ± 118.8 | 570.0 ± 127.5 |
| Success rate | 0.162 ± 0.042 | 0.661 ± 0.032 | 0.865 ± 0.018 |
| Utilization | 0.673 ± 0.040 | 0.671 ± 0.039 | 0.669 ± 0.035 |
| Avg slowdown | 68.78 ± 16.74 | 87.57 ± 19.45 | 76.78 ± 20.66 |

### kNN-TEWMA-CA minus the other schedulers, same seed

| Metric | vs TEWMA: mean diff [95% CI] | relative | better in | vs Baseline: mean diff [95% CI] | relative | better in |
|:--|--:|--:|--:|--:|--:|--:|
| Makespan | +107.1 [+84.7, +129.5] | +7.7% | 1/30 | +318.5 [+290.0, +347.0] | +26.7% | 0/30 |
| Avg wait | -2.5 [-20.3, +15.3] | -0.2% | 15/30 | +98.6 [+76.1, +121.2] | +23.3% | 1/30 |
| Avg turnaround | -1.1 [-18.8, +16.7] | +0.1% | 15/30 | +102.9 [+80.4, +125.5] | +23.3% | 1/30 |
| Success rate | +0.204 [+0.192, +0.216] | +31.1% | 30/30 | +0.702 [+0.684, +0.721] | +466.5% | 30/30 |
| Utilization | -0.002 [-0.012, +0.008] | -0.2% | 14/30 | -0.004 [-0.017, +0.008] | -0.5% | 14/30 |
| Avg slowdown | -10.79 [-15.90, -5.68] | -11.5% | 23/30 | +8.00 [+2.82, +13.19] | +13.1% | 8/30 |

## Reservation guarantee

Reservation violations are not reported as a result. Every backfilled job is allocated a runtime that ends by the head job's reservation and is killed at its allocation, so backfilling cannot delay a reservation under any of the three schedulers. The simulator re-checks this after every backfill round and raises `ReservationGuaranteeError` if it ever fails. It held in all 690 simulations above, as it must.

