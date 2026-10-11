# Round 2 presentation, reproduced

Reservations planned from true finish times, one seed per experiment (E1 and E2: 42, E3: 99, E4: 7). These are the numbers in the slides and the write-up, produced by `knn-tewma-ca reproduce`.

## E1 · Baseline comparison (300 jobs, 8 nodes, p = 0.80)

| Metric | Baseline | TEWMA | kNN-TEWMA-CA | CA vs Baseline |
|:--|--:|--:|--:|--:|
| Makespan | 2202.20 | 2197.99 | 2211.59 | +0.4% |
| Avg wait | 70.66 | 74.37 | 80.43 | +13.8% |
| Avg turnaround | 89.67 | 94.08 | 100.59 | +12.2% |
| Success rate | 0.5400 | 0.7233 | 0.8433 | +56.2% |
| Utilization | 0.7600 | 0.7913 | 0.8054 | +6.0% |
| Avg slowdown | 11.46 | 12.18 | 11.69 | +2.0% |

Reservation violations (kNN-TEWMA-CA): 0. The slides list this as a result; it is guaranteed by the backfill rule for all three schedulers (see the README).

## E2 · Confidence sweep (kNN-TEWMA-CA, 300 jobs)

| Confidence | Success rate | Utilization | Avg slowdown |
|:--|--:|--:|--:|
| p70 | 0.8100 | 0.7876 | 10.74 |
| p75 | 0.8267 | 0.7968 | 11.37 |
| p80 | 0.8433 | 0.8054 | 11.69 |
| p85 | 0.8633 | 0.8070 | 11.89 |
| p90 | 0.8667 | 0.8072 | 11.89 |

## E3 · Load sensitivity (kNN-TEWMA-CA, p = 0.80)

| Jobs | Success rate | Avg wait | Avg slowdown |
|:--|--:|--:|--:|
| 50 | 0.8000 | 14.33 | 3.22 |
| 100 | 0.8400 | 65.78 | 9.53 |
| 200 | 0.8500 | 98.43 | 15.28 |
| 400 | 0.8425 | 193.32 | 26.95 |

## E4 · High-contention stress (150 jobs, λ = 1.2, bias = −0.25, noise × 0.8)

| Metric | TEWMA | kNN-TEWMA-CA | Δ |
|:--|--:|--:|--:|
| Makespan | 1528.65 | 1632.48 | +6.8% |
| Avg wait | 545.61 | 588.64 | +7.9% |
| Avg turnaround | 566.39 | 610.48 | +7.8% |
| Success rate | 0.7067 | 0.8933 | +26.4% |
| Utilization | 0.6711 | 0.6584 | -1.9% |
| Avg slowdown | 93.45 | 102.88 | +10.1% |

