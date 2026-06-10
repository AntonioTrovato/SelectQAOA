# Dunn Test Pairwise Comparisons with A12 Effect Size

This file reports, for each program, the pairwise Dunn test results and the corresponding A12 effect size.

## flex

| Pair | p-value | adj. p-value | A12 |
|---|---:|---:|---:|
| `selectqa` vs `div_ga` | 4.728114e-08 | 1.323872e-06 | 0.00 |
| `selectqa` vs `add_greedy` | 5.249718e-01 | 1.000000e+00 | 0.00 |
| `selectqa` vs `qaoa_statevector_sim` | 1.330951e-02 | 3.726662e-01 | 1.00 |
| `selectqa` vs `qaoa_aer_sim` | 4.079622e-04 | 1.142294e-02 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_1` | 1.625296e-01 | 1.000000e+00 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_2` | 6.857599e-06 | 1.920128e-04 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_5` | 2.001823e-01 | 1.000000e+00 | 1.00 |
| `div_ga` vs `add_greedy` | 1.396223e-06 | 3.909424e-05 | 1.00 |
| `div_ga` vs `qaoa_statevector_sim` | 2.827775e-03 | 7.917769e-02 | 1.00 |
| `div_ga` vs `qaoa_aer_sim` | 5.405888e-02 | 1.000000e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_1` | 4.810891e-05 | 1.347049e-03 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_2` | 3.354561e-01 | 1.000000e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_5` | 1.559248e-11 | 4.365895e-10 | 1.00 |
| `add_greedy` vs `qaoa_statevector_sim` | 6.581551e-02 | 1.000000e+00 | 1.00 |
| `add_greedy` vs `qaoa_aer_sim` | 3.741418e-03 | 1.047597e-01 | 1.00 |
| `add_greedy` vs `qaoa_depolarizing_1` | 4.467085e-01 | 1.000000e+00 | 1.00 |
| `add_greedy` vs `qaoa_depolarizing_2` | 1.122965e-04 | 3.144303e-03 | 1.00 |
| `add_greedy` vs `qaoa_depolarizing_5` | 5.527190e-02 | 1.000000e+00 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_aer_sim` | 2.893719e-01 | 1.000000e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_1` | 2.806929e-01 | 1.000000e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_2` | 4.310587e-02 | 1.000000e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_5` | 1.723677e-04 | 4.826295e-03 | 0.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_1` | 3.249527e-02 | 9.098677e-01 | 0.99 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_2` | 3.354561e-01 | 1.000000e+00 | 0.93 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_5` | 1.465297e-06 | 4.102830e-05 | 0.98 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_2` | 1.925775e-03 | 5.392169e-02 | 0.08 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_5` | 7.414139e-03 | 2.075959e-01 | 0.24 |
| `qaoa_depolarizing_2` vs `qaoa_depolarizing_5` | 7.510419e-09 | 2.102917e-07 | 0.79 |

## grep

| Pair | p-value | adj. p-value | A12 |
|---|---:|---:|---:|
| `selectqa` vs `div_ga` | 2.317489e-08 | 6.488968e-07 | 0.88 |
| `selectqa` vs `add_greedy` | 2.893719e-01 | 1.000000e+00 | 1.00 |
| `selectqa` vs `qaoa_statevector_sim` | 2.597749e-01 | 1.000000e+00 | 1.00 |
| `selectqa` vs `qaoa_aer_sim` | 2.641561e-10 | 7.396371e-09 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_1` | 1.828496e-02 | 5.119789e-01 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_2` | 6.553634e-06 | 1.835017e-04 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_5` | 3.933426e-04 | 1.101359e-02 | 1.00 |
| `div_ga` vs `add_greedy` | 5.983924e-06 | 1.675499e-04 | 1.00 |
| `div_ga` vs `qaoa_statevector_sim` | 8.213812e-06 | 2.299867e-04 | 1.00 |
| `div_ga` vs `qaoa_aer_sim` | 4.641573e-01 | 1.000000e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_1` | 1.252440e-03 | 3.506831e-02 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_2` | 2.806929e-01 | 1.000000e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_5` | 4.115685e-02 | 1.000000e+00 | 1.00 |
| `add_greedy` vs `qaoa_statevector_sim` | 9.462452e-01 | 1.000000e+00 | 1.00 |
| `add_greedy` vs `qaoa_aer_sim` | 1.448605e-07 | 4.056095e-06 | 1.00 |
| `add_greedy` vs `qaoa_depolarizing_1` | 1.934995e-01 | 1.000000e+00 | 1.00 |
| `add_greedy` vs `qaoa_depolarizing_2` | 5.643518e-04 | 1.580185e-02 | 1.00 |
| `add_greedy` vs `qaoa_depolarizing_5` | 1.295477e-02 | 3.627336e-01 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_aer_sim` | 2.085480e-07 | 5.839343e-06 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_1` | 2.176230e-01 | 1.000000e+00 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_2` | 7.228310e-04 | 2.023927e-02 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_5` | 1.562378e-02 | 4.374658e-01 | 1.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_1` | 7.536544e-05 | 2.110232e-03 | 0.88 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_2` | 7.017451e-02 | 1.000000e+00 | 0.05 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_5` | 5.537766e-03 | 1.550574e-01 | 0.56 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_2` | 3.172199e-02 | 8.882156e-01 | 0.10 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_5` | 2.361305e-01 | 1.000000e+00 | 0.10 |
| `qaoa_depolarizing_2` vs `qaoa_depolarizing_5` | 3.354561e-01 | 1.000000e+00 | 0.92 |

## gzip

| Pair | p-value | adj. p-value | A12 |
|---|---:|---:|---:|
| `selectqa` vs `div_ga` | 1.862928e-06 | 5.216198e-05 | 0.00 |
| `selectqa` vs `add_greedy` | 2.361305e-01 | 1.000000e+00 | 1.00 |
| `selectqa` vs `qaoa_statevector_sim` | 4.021067e-02 | 1.000000e+00 | 1.00 |
| `selectqa` vs `qaoa_aer_sim` | 1.420508e-04 | 3.977423e-03 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_1` | 1.876511e-02 | 5.254230e-01 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_2` | 2.285982e-01 | 1.000000e+00 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_5` | 7.476764e-02 | 1.000000e+00 | 1.00 |
| `div_ga` vs `add_greedy` | 3.396212e-04 | 9.509395e-03 | 1.00 |
| `div_ga` vs `qaoa_statevector_sim` | 6.604121e-03 | 1.849154e-01 | 1.00 |
| `div_ga` vs `qaoa_aer_sim` | 3.354561e-01 | 1.000000e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_1` | 1.562378e-02 | 4.374658e-01 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_2` | 2.347509e-09 | 6.573026e-08 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_5` | 5.767623e-11 | 1.614934e-09 | 1.00 |
| `add_greedy` vs `qaoa_statevector_sim` | 3.860165e-01 | 1.000000e+00 | 0.80 |
| `add_greedy` vs `qaoa_aer_sim` | 8.796698e-03 | 2.463075e-01 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_1` | 2.438367e-01 | 1.000000e+00 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_2` | 1.690847e-02 | 4.734372e-01 | 0.00 |
| `add_greedy` vs `qaoa_depolarizing_5` | 3.011111e-03 | 8.431112e-02 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_aer_sim` | 7.960343e-02 | 1.000000e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_1` | 7.652556e-01 | 1.000000e+00 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_2` | 1.131706e-03 | 3.168776e-02 | 0.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_5` | 1.263506e-04 | 3.537817e-03 | 0.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_1` | 1.458340e-01 | 1.000000e+00 | 0.97 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_2` | 5.484229e-07 | 1.535584e-05 | 0.45 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_5` | 2.317489e-08 | 6.488968e-07 | 0.76 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_2` | 3.792137e-04 | 1.061798e-02 | 0.00 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_5` | 3.595393e-05 | 1.006710e-03 | 0.02 |
| `qaoa_depolarizing_2` vs `qaoa_depolarizing_5` | 5.633253e-01 | 1.000000e+00 | 0.94 |

## sed

| Pair | p-value | adj. p-value | A12 |
|---|---:|---:|---:|
| `selectqa` vs `div_ga` | 2.317385e-11 | 6.488677e-10 | 0.00 |
| `selectqa` vs `add_greedy` | 2.399618e-01 | 1.000000e+00 | 1.00 |
| `selectqa` vs `qaoa_statevector_sim` | 2.481859e-02 | 6.949207e-01 | 1.00 |
| `selectqa` vs `qaoa_aer_sim` | 1.057105e-08 | 2.959893e-07 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_1` | 3.096453e-02 | 8.670068e-01 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_2` | 1.954040e-06 | 5.471313e-05 | 1.00 |
| `selectqa` vs `qaoa_depolarizing_5` | 1.658555e-04 | 4.643954e-03 | 1.00 |
| `div_ga` vs `add_greedy` | 3.600539e-08 | 1.008151e-06 | 1.00 |
| `div_ga` vs `qaoa_statevector_sim` | 8.984602e-06 | 2.515688e-04 | 1.00 |
| `div_ga` vs `qaoa_aer_sim` | 3.354561e-01 | 1.000000e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_1` | 5.983924e-06 | 1.675499e-04 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_2` | 5.405888e-02 | 1.000000e+00 | 1.00 |
| `div_ga` vs `qaoa_depolarizing_5` | 3.517851e-03 | 9.849982e-02 | 1.00 |
| `add_greedy` vs `qaoa_statevector_sim` | 2.850101e-01 | 1.000000e+00 | 1.00 |
| `add_greedy` vs `qaoa_aer_sim` | 5.461789e-06 | 1.529301e-04 | 1.00 |
| `add_greedy` vs `qaoa_depolarizing_1` | 3.258802e-01 | 1.000000e+00 | 1.00 |
| `add_greedy` vs `qaoa_depolarizing_2` | 3.396212e-04 | 9.509395e-03 | 1.00 |
| `add_greedy` vs `qaoa_depolarizing_5` | 9.570855e-03 | 2.679839e-01 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_aer_sim` | 5.068916e-04 | 1.419296e-02 | 0.99 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_1` | 9.309208e-01 | 1.000000e+00 | 1.00 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_2` | 1.194024e-02 | 3.343266e-01 | 0.98 |
| `qaoa_statevector_sim` vs `qaoa_depolarizing_5` | 1.280522e-01 | 1.000000e+00 | 1.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_1` | 3.655603e-04 | 1.023569e-02 | 1.00 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_2` | 3.354561e-01 | 1.000000e+00 | 0.54 |
| `qaoa_aer_sim` vs `qaoa_depolarizing_5` | 5.055239e-02 | 1.000000e+00 | 0.84 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_2` | 9.306304e-03 | 2.605765e-01 | 0.00 |
| `qaoa_depolarizing_1` vs `qaoa_depolarizing_5` | 1.077226e-01 | 1.000000e+00 | 0.06 |
| `qaoa_depolarizing_2` vs `qaoa_depolarizing_5` | 3.211596e-01 | 1.000000e+00 | 0.82 |
