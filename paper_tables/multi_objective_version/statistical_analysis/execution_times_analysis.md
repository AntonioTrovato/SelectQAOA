# Dunn Test Pairwise Comparisons with A12 Effect Size — Multi-Objective Efficiency

This file reports, for each program, the pairwise Dunn test results 
(Bonferroni-corrected) and the corresponding A12 effect size.
A12 > 0.5 means Group1 tends to be slower than Group2.

## flex

| Pair | p-value | adj. p-value | A12 |
|---|---:|---:|---:|
| `qtcs` vs `div_ga` | 3.92e-01 | 1.00e+00 | 0.00 |
| `qtcs` vs `add_greedy` | 5.93e-04 | 2.14e-02 | 0.00 |
| `qtcs` vs `qaoa_statevector_sim` | 7.21e-02 | 1.00e+00 | 0.00 |
| `qtcs` vs `qaoa_aer_sim` | 7.28e-12 | 2.62e-10 | 0.00 |
| `qtcs` vs `qaoa_fake_brisbane` | 2.76e-07 | 9.94e-06 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_1` | 5.82e-03 | 2.09e-01 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_2` | 2.03e-09 | 7.30e-08 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_5` | 6.34e-05 | 2.28e-03 | 0.00 |
| `div_ga` vs `add_greedy` | 9.94e-03 | 3.58e-01 | 1.00 |
| `div_ga` vs `qaoa_statevector_sim` | 3.46e-01 | 1.00e+00 | 1.00 |
| `div_ga` vs `qaoa_aer_sim` | 2.03e-09 | 7.30e-08 | 1.00 |
| `div_ga` vs `qaoa_fake_brisbane` | 1.85e-05 | 6.65e-04 | 0.00 |
| `div_ga` vs `qaoa_depolarizing_1` | 5.72e-02 | 1.00e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_2` | 2.76e-07 | 9.94e-06 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_5` | 1.67e-03 | 6.01e-02 | 1.00 |
| `add_greedy` vs `qaoa_statevector_sim` | 1.02e-01 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_aer_sim` | 6.32e-04 | 2.28e-02 | 0.00 |
| `add_greedy` vs `qaoa_fake_brisbane` | 8.83e-02 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_1` | 4.99e-01 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_2` | 1.04e-02 | 3.76e-01 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_5` | 5.72e-01 | 1.00e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_aer_sim` | 4.34e-07 | 1.56e-05 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_fake_brisbane` | 8.37e-04 | 3.01e-02 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_1` | 3.37e-01 | 1.00e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_2` | 2.71e-05 | 9.74e-04 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_5` | 2.77e-02 | 1.00e+00 | 0.00 |
| `qaoa_aer_sim` vs `qaoa_fake_brisbane` | 8.67e-02 | 1.00e+00 | 0.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_1` | 4.24e-05 | 1.53e-03 | 0.99 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_2` | 3.92e-01 | 1.00e+00 | 0.93 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_5` | 4.34e-03 | 1.56e-01 | 0.98 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_1` | 1.73e-02 | 6.21e-01 | 1.00 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_2` | 2.76e-07 | 9.94e-06 | 1.00 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_5` | 2.55e-01 | 1.00e+00 | 1.00 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_2` | 1.21e-03 | 4.34e-02 | 0.08 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_5` | 2.14e-01 | 1.00e+00 | 0.24 |
| `qaoa_depolarizing_2` vs `qaoa_depolarizing_5` | 4.60e-02 | 1.00e+00 | 0.79 |

## grep

| Pair | p-value | adj. p-value | A12 |
|---|---:|---:|---:|
| `qtcs` vs `div_ga` | 1.21e-01 | 1.00e+00 | 0.00 |
| `qtcs` vs `add_greedy` | 8.75e-05 | 3.15e-03 | 0.00 |
| `qtcs` vs `qaoa_statevector_sim` | 1.12e-04 | 4.03e-03 | 0.00 |
| `qtcs` vs `qaoa_aer_sim` | 7.28e-12 | 2.62e-10 | 0.00 |
| `qtcs` vs `qaoa_fake_brisbane` | 3.08e-01 | 1.00e+00 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_1` | 5.67e-03 | 2.04e-01 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_2` | 2.03e-09 | 7.30e-08 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_5` | 1.14e-06 | 4.12e-05 | 0.00 |
| `div_ga` vs `add_greedy` | 1.77e-02 | 6.36e-01 | 1.00 |
| `div_ga` vs `qaoa_statevector_sim` | 2.07e-02 | 7.47e-01 | 0.19 |
| `div_ga` vs `qaoa_aer_sim` | 1.15e-07 | 4.13e-06 | 1.00 |
| `div_ga` vs `qaoa_fake_brisbane` | 5.95e-01 | 1.00e+00 | 0.00 |
| `div_ga` vs `qaoa_depolarizing_1` | 2.24e-01 | 1.00e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_2` | 8.78e-06 | 3.16e-04 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_5` | 9.18e-04 | 3.30e-02 | 1.00 |
| `add_greedy` vs `qaoa_statevector_sim` | 9.52e-01 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_aer_sim` | 3.40e-03 | 1.22e-01 | 0.00 |
| `add_greedy` vs `qaoa_fake_brisbane` | 3.69e-03 | 1.33e-01 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_1` | 2.48e-01 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_2` | 3.82e-02 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_5` | 3.46e-01 | 1.00e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_aer_sim` | 2.80e-03 | 1.01e-01 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_fake_brisbane` | 4.46e-03 | 1.61e-01 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_1` | 2.73e-01 | 1.00e+00 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_2` | 3.30e-02 | 1.00e+00 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_5` | 3.16e-01 | 1.00e+00 | 1.00 |
| `qaoa_aer_sim` vs `qaoa_fake_brisbane` | 5.45e-09 | 1.96e-07 | 0.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_1` | 4.40e-05 | 1.58e-03 | 0.88 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_2` | 3.92e-01 | 1.00e+00 | 0.05 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_5` | 4.69e-02 | 1.00e+00 | 0.56 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_1` | 8.06e-02 | 1.00e+00 | 1.00 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_2` | 6.48e-07 | 2.33e-05 | 1.00 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_5` | 1.20e-04 | 4.33e-03 | 1.00 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_2` | 1.24e-03 | 4.47e-02 | 0.10 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_5` | 3.59e-02 | 1.00e+00 | 0.10 |
| `qaoa_depolarizing_2` vs `qaoa_depolarizing_5` | 2.58e-01 | 1.00e+00 | 0.92 |

## gzip

| Pair | p-value | adj. p-value | A12 |
|---|---:|---:|---:|
| `qtcs` vs `div_ga` | 2.76e-07 | 9.94e-06 | 0.00 |
| `qtcs` vs `add_greedy` | 1.98e-02 | 7.14e-01 | 1.00 |
| `qtcs` vs `qaoa_statevector_sim` | 1.19e-01 | 1.00e+00 | 0.00 |
| `qtcs` vs `qaoa_aer_sim` | 2.03e-09 | 7.30e-08 | 0.00 |
| `qtcs` vs `qaoa_fake_brisbane` | 1.85e-05 | 6.65e-04 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_1` | 1.96e-01 | 1.00e+00 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_2` | 7.28e-12 | 2.62e-10 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_5` | 7.17e-04 | 2.58e-02 | 0.00 |
| `div_ga` vs `add_greedy` | 4.96e-03 | 1.79e-01 | 1.00 |
| `div_ga` vs `qaoa_statevector_sim` | 3.43e-04 | 1.24e-02 | 1.00 |
| `div_ga` vs `qaoa_aer_sim` | 3.92e-01 | 1.00e+00 | 1.00 |
| `div_ga` vs `qaoa_fake_brisbane` | 3.92e-01 | 1.00e+00 | 0.00 |
| `div_ga` vs `qaoa_depolarizing_1` | 1.20e-04 | 4.33e-03 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_2` | 8.67e-02 | 1.00e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_5` | 7.91e-02 | 1.00e+00 | 1.00 |
| `add_greedy` vs `qaoa_statevector_sim` | 4.41e-01 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_aer_sim` | 2.47e-04 | 8.88e-03 | 0.00 |
| `add_greedy` vs `qaoa_fake_brisbane` | 5.08e-02 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_1` | 3.00e-01 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_2` | 6.12e-06 | 2.20e-04 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_5` | 2.92e-01 | 1.00e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_aer_sim` | 9.14e-06 | 3.29e-04 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_fake_brisbane` | 6.52e-03 | 2.35e-01 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_1` | 7.91e-01 | 1.00e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_2` | 1.20e-07 | 4.33e-06 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_5` | 6.81e-02 | 1.00e+00 | 0.00 |
| `qaoa_aer_sim` vs `qaoa_fake_brisbane` | 8.67e-02 | 1.00e+00 | 0.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_1` | 2.57e-06 | 9.27e-05 | 0.97 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_2` | 3.92e-01 | 1.00e+00 | 0.45 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_5` | 8.99e-03 | 3.24e-01 | 0.76 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_1` | 2.80e-03 | 1.01e-01 | 1.00 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_2` | 1.02e-02 | 3.67e-01 | 1.00 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_5` | 3.68e-01 | 1.00e+00 | 1.00 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_2` | 2.72e-08 | 9.78e-07 | 0.00 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_5` | 3.66e-02 | 1.00e+00 | 0.02 |
| `qaoa_depolarizing_2` vs `qaoa_depolarizing_5` | 5.23e-04 | 1.88e-02 | 0.94 |

## sed

| Pair | p-value | adj. p-value | A12 |
|---|---:|---:|---:|
| `qtcs` vs `div_ga` | 3.92e-01 | 1.00e+00 | 0.00 |
| `qtcs` vs `add_greedy` | 5.28e-05 | 1.90e-03 | 1.00 |
| `qtcs` vs `qaoa_statevector_sim` | 1.99e-03 | 7.16e-02 | 0.00 |
| `qtcs` vs `qaoa_aer_sim` | 2.03e-09 | 7.30e-08 | 0.00 |
| `qtcs` vs `qaoa_fake_brisbane` | 8.21e-02 | 1.00e+00 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_1` | 1.53e-03 | 5.51e-02 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_2` | 7.28e-12 | 2.62e-10 | 0.00 |
| `qtcs` vs `qaoa_depolarizing_5` | 3.63e-07 | 1.31e-05 | 0.00 |
| `div_ga` vs `add_greedy` | 1.44e-03 | 5.19e-02 | 1.00 |
| `div_ga` vs `qaoa_statevector_sim` | 2.54e-02 | 9.14e-01 | 1.00 |
| `div_ga` vs `qaoa_aer_sim` | 2.76e-07 | 9.94e-06 | 1.00 |
| `div_ga` vs `qaoa_fake_brisbane` | 3.78e-01 | 1.00e+00 | 0.00 |
| `div_ga` vs `qaoa_depolarizing_1` | 2.07e-02 | 7.47e-01 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_2` | 5.41e-02 | 1.00e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_5` | 2.33e-05 | 8.37e-04 | 1.00 |
| `add_greedy` vs `qaoa_statevector_sim` | 3.42e-01 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_aer_sim` | 5.08e-02 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_fake_brisbane` | 2.12e-02 | 7.64e-01 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_1` | 3.82e-01 | 1.00e+00 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_2` | 4.96e-03 | 1.79e-01 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_5` | 2.96e-01 | 1.00e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_aer_sim` | 3.69e-03 | 1.33e-01 | 0.99 |
| `qaoa_statevector_sim` vs `qaoa_fake_brisbane` | 1.76e-01 | 1.00e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_1` | 9.39e-01 | 1.00e+00 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_2` | 1.70e-04 | 6.12e-03 | 0.98 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_5` | 4.60e-02 | 1.00e+00 | 1.00 |
| `qaoa_aer_sim` vs `qaoa_fake_brisbane` | 2.07e-05 | 7.46e-04 | 0.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_1` | 4.71e-03 | 1.69e-01 | 1.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_2` | 3.92e-01 | 1.00e+00 | 0.54 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_5` | 5.06e-02 | 1.00e+00 | 0.84 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_1` | 1.53e-01 | 1.00e+00 | 1.00 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_2` | 3.17e-07 | 1.14e-05 | 1.00 |
| `qaoa_fake_brisbane` vs `qaoa_depolarizing_5` | 8.11e-04 | 2.92e-02 | 1.00 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_2` | 2.31e-04 | 8.30e-03 | 0.00 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_5` | 5.50e-02 | 1.00e+00 | 0.06 |
| `qaoa_depolarizing_2` vs `qaoa_depolarizing_5` | 7.77e-02 | 1.00e+00 | 0.82 |