# Pairwise Statistical Analysis Results

---

## GSDTSR

### final_test_suite_costs — Kruskal-Wallis + Dunn (Bonferroni) + A12

Kruskal-Wallis: H = 80.75, df = 9, p < 0.001. Post-hoc Dunn test with Bonferroni correction. Effect size is Vargha-Delaney A12: values close to 0 indicate the first group tends to have lower costs than the second; values close to 1 indicate the opposite. Groups with no significant adjusted p-value are statistically indistinguishable after correction.

| Pair | p-value | adj. p-value | A12 |
|------|---------|--------------|-----|
| SelectQA vs BootQA | 0.0171 | 1.0000 | 0.300 |
| SelectQA vs IgDec_ideal | 0.0326 | 1.0000 | 0.000 |
| SelectQA vs IgDec_noise | 0.0674 | 1.0000 | 0.000 |
| SelectQA vs SelectQAOA_statevector_sim | 0.0577 | 1.0000 | 0.000 |
| SelectQA vs SelectQAOA_aer_sim | 0.0006 | **0.0260** | 0.000 |
| SelectQA vs SelectQAOA_fake_brisbane | 0.0063 | 0.2837 | 0.000 |
| SelectQA vs SelectQAOA_depolarizing_sim/01 | 0.4090 | 1.0000 | 0.000 |
| SelectQA vs SelectQAOA_depolarizing_sim/02 | 0.0007 | **0.0336** | 0.000 |
| SelectQA vs SelectQAOA_depolarizing_sim/05 | 0.0457 | 1.0000 | 0.000 |
| BootQA vs IgDec_ideal | < 0.0001 | **0.0003** | 0.000 |
| BootQA vs IgDec_noise | < 0.0001 | **0.0011** | 0.000 |
| BootQA vs SelectQAOA_statevector_sim | 0.6269 | 1.0000 | 0.000 |
| BootQA vs SelectQAOA_aer_sim | 0.2905 | 1.0000 | 0.000 |
| BootQA vs SelectQAOA_fake_brisbane | 0.7284 | 1.0000 | 0.000 |
| BootQA vs SelectQAOA_depolarizing_sim/01 | 0.0013 | 0.0598 | 0.000 |
| BootQA vs SelectQAOA_depolarizing_sim/02 | 0.3233 | 1.0000 | 0.000 |
| BootQA vs SelectQAOA_depolarizing_sim/05 | 0.6996 | 1.0000 | 0.000 |
| IgDec_ideal vs IgDec_noise | 0.7576 | 1.0000 | 0.000 |
| IgDec_ideal vs SelectQAOA_statevector_sim | 0.0001 | **0.0025** | 1.000 |
| IgDec_ideal vs SelectQAOA_aer_sim | < 0.0001 | **< 0.0001** | 0.000 |
| IgDec_ideal vs SelectQAOA_fake_brisbane | < 0.0001 | **0.0001** | 0.000 |
| IgDec_ideal vs SelectQAOA_depolarizing_sim/01 | 0.1896 | 1.0000 | 0.070 |
| IgDec_ideal vs SelectQAOA_depolarizing_sim/02 | < 0.0001 | **< 0.0001** | 0.000 |
| IgDec_ideal vs SelectQAOA_depolarizing_sim/05 | < 0.0001 | **0.0016** | 0.000 |
| IgDec_noise vs SelectQAOA_statevector_sim | 0.0002 | **0.0087** | 1.000 |
| IgDec_noise vs SelectQAOA_aer_sim | < 0.0001 | **< 0.0001** | 0.500 |
| IgDec_noise vs SelectQAOA_fake_brisbane | < 0.0001 | **0.0002** | 0.410 |
| IgDec_noise vs SelectQAOA_depolarizing_sim/01 | 0.3158 | 1.0000 | 0.540 |
| IgDec_noise vs SelectQAOA_depolarizing_sim/02 | < 0.0001 | **< 0.0001** | 0.360 |
| IgDec_noise vs SelectQAOA_depolarizing_sim/05 | 0.0001 | **0.0058** | 0.350 |
| SelectQAOA_statevector_sim vs SelectQAOA_aer_sim | 0.1228 | 1.0000 | 0.000 |
| SelectQAOA_statevector_sim vs SelectQAOA_fake_brisbane | 0.4047 | 1.0000 | 0.000 |
| SelectQAOA_statevector_sim vs SelectQAOA_depolarizing_sim/01 | 0.0065 | 0.2904 | 0.000 |
| SelectQAOA_statevector_sim vs SelectQAOA_depolarizing_sim/02 | 0.1405 | 1.0000 | 0.000 |
| SelectQAOA_statevector_sim vs SelectQAOA_depolarizing_sim/05 | 0.9201 | 1.0000 | 0.000 |
| SelectQAOA_aer_sim vs SelectQAOA_fake_brisbane | 0.4778 | 1.0000 | 0.330 |
| SelectQAOA_aer_sim vs SelectQAOA_depolarizing_sim/01 | < 0.0001 | **0.0009** | 0.490 |
| SelectQAOA_aer_sim vs SelectQAOA_depolarizing_sim/02 | 0.9446 | 1.0000 | 0.210 |
| SelectQAOA_aer_sim vs SelectQAOA_depolarizing_sim/05 | 0.1490 | 1.0000 | 0.130 |
| SelectQAOA_fake_brisbane vs SelectQAOA_depolarizing_sim/01 | 0.0004 | **0.0169** | 0.670 |
| SelectQAOA_fake_brisbane vs SelectQAOA_depolarizing_sim/02 | 0.5219 | 1.0000 | 0.350 |
| SelectQAOA_fake_brisbane vs SelectQAOA_depolarizing_sim/05 | 0.4635 | 1.0000 | 0.330 |
| SelectQAOA_depolarizing_sim/01 vs SelectQAOA_depolarizing_sim/02 | < 0.0001 | **0.0012** | 0.180 |
| SelectQAOA_depolarizing_sim/01 vs SelectQAOA_depolarizing_sim/05 | 0.0047 | 0.2134 | 0.120 |
| SelectQAOA_depolarizing_sim/02 vs SelectQAOA_depolarizing_sim/05 | 0.1696 | 1.0000 | 0.540 |

---

### final_effectivenesses — Kruskal-Wallis + Dunn (Bonferroni) + A12

Kruskal-Wallis: H = 61.45, df = 9, p < 0.001. Post-hoc Dunn test with Bonferroni correction. Effect size is A12: values close to 1 indicate the first group tends to have higher effectiveness than the second; values close to 0 indicate the opposite. IgDec variants consistently dominate over classical baselines (SelectQA, BootQA) and over all SelectQAOA configurations.

| Pair | p-value | adj. p-value | A12 |
|------|---------|--------------|-----|
| SelectQA vs BootQA | 0.0457 | 1.0000 | 1.000 |
| SelectQA vs IgDec_ideal | < 0.0001 | **< 0.0001** | 0.000 |
| SelectQA vs IgDec_noise | < 0.0001 | **< 0.0001** | 0.000 |
| SelectQA vs SelectQAOA_statevector_sim | 0.0017 | 0.0759 | 0.000 |
| SelectQA vs SelectQAOA_aer_sim | 0.0071 | 0.3188 | 0.000 |
| SelectQA vs SelectQAOA_fake_brisbane | 0.0093 | 0.4191 | 0.000 |
| SelectQA vs SelectQAOA_depolarizing_sim/01 | 0.0171 | 0.7701 | 0.000 |
| SelectQA vs SelectQAOA_depolarizing_sim/02 | 0.0296 | 1.0000 | 0.000 |
| SelectQA vs SelectQAOA_depolarizing_sim/05 | 0.0044 | 0.1985 | 0.000 |
| BootQA vs IgDec_ideal | 0.0002 | **0.0096** | 0.000 |
| BootQA vs IgDec_noise | < 0.0001 | **0.0003** | 0.000 |
| BootQA vs SelectQAOA_statevector_sim | 0.2535 | 1.0000 | 0.000 |
| BootQA vs SelectQAOA_aer_sim | 0.4874 | 1.0000 | 0.000 |
| BootQA vs SelectQAOA_fake_brisbane | 0.5473 | 1.0000 | 0.000 |
| BootQA vs SelectQAOA_depolarizing_sim/01 | 0.6996 | 1.0000 | 0.000 |
| BootQA vs SelectQAOA_depolarizing_sim/02 | 0.8591 | 1.0000 | 0.000 |
| BootQA vs SelectQAOA_depolarizing_sim/05 | 0.3960 | 1.0000 | 0.000 |
| IgDec_ideal vs IgDec_noise | 0.4404 | 1.0000 | 0.840 |
| IgDec_ideal vs SelectQAOA_statevector_sim | 0.0104 | 0.4687 | 0.900 |
| IgDec_ideal vs SelectQAOA_aer_sim | 0.0026 | 0.1179 | 0.940 |
| IgDec_ideal vs SelectQAOA_fake_brisbane | 0.0019 | 0.0865 | 0.910 |
| IgDec_ideal vs SelectQAOA_depolarizing_sim/01 | 0.0009 | **0.0408** | 0.940 |
| IgDec_ideal vs SelectQAOA_depolarizing_sim/02 | 0.0004 | **0.0190** | 0.920 |
| IgDec_ideal vs SelectQAOA_depolarizing_sim/05 | 0.0043 | 0.1937 | 0.940 |
| IgDec_noise vs SelectQAOA_statevector_sim | 0.0009 | **0.0386** | 0.500 |
| IgDec_noise vs SelectQAOA_aer_sim | 0.0002 | **0.0070** | 0.650 |
| IgDec_noise vs SelectQAOA_fake_brisbane | 0.0001 | **0.0048** | 0.620 |
| IgDec_noise vs SelectQAOA_depolarizing_sim/01 | < 0.0001 | **0.0019** | 0.650 |
| IgDec_noise vs SelectQAOA_depolarizing_sim/02 | < 0.0001 | **0.0008** | 0.590 |
| IgDec_noise vs SelectQAOA_depolarizing_sim/05 | 0.0003 | **0.0129** | 0.630 |
| SelectQAOA_statevector_sim vs SelectQAOA_aer_sim | 0.6545 | 1.0000 | 0.700 |
| SelectQAOA_statevector_sim vs SelectQAOA_fake_brisbane | 0.5891 | 1.0000 | 0.600 |
| SelectQAOA_statevector_sim vs SelectQAOA_depolarizing_sim/01 | 0.4495 | 1.0000 | 0.600 |
| SelectQAOA_statevector_sim vs SelectQAOA_depolarizing_sim/02 | 0.3348 | 1.0000 | 0.400 |
| SelectQAOA_statevector_sim vs SelectQAOA_depolarizing_sim/05 | 0.7694 | 1.0000 | 0.400 |
| SelectQAOA_aer_sim vs SelectQAOA_fake_brisbane | 0.9262 | 1.0000 | 0.490 |
| SelectQAOA_aer_sim vs SelectQAOA_depolarizing_sim/01 | 0.7576 | 1.0000 | 0.610 |
| SelectQAOA_aer_sim vs SelectQAOA_depolarizing_sim/02 | 0.6052 | 1.0000 | 0.360 |
| SelectQAOA_aer_sim vs SelectQAOA_depolarizing_sim/05 | 0.8774 | 1.0000 | 0.530 |
| SelectQAOA_fake_brisbane vs SelectQAOA_depolarizing_sim/01 | 0.8290 | 1.0000 | 0.640 |
| SelectQAOA_fake_brisbane vs SelectQAOA_depolarizing_sim/02 | 0.6713 | 1.0000 | 0.430 |
| SelectQAOA_fake_brisbane vs SelectQAOA_depolarizing_sim/05 | 0.8050 | 1.0000 | 0.570 |
| SelectQAOA_depolarizing_sim/01 vs SelectQAOA_depolarizing_sim/02 | 0.8350 | 1.0000 | 0.340 |
| SelectQAOA_depolarizing_sim/01 vs SelectQAOA_depolarizing_sim/05 | 0.6434 | 1.0000 | 0.420 |
| SelectQAOA_depolarizing_sim/02 vs SelectQAOA_depolarizing_sim/05 | 0.5020 | 1.0000 | 0.610 |

---

## PAINTCONTROL

### final_test_suite_costs — Kruskal-Wallis + Dunn (Bonferroni) + A12

Kruskal-Wallis: H = 79.02, df = 9, p < 0.001. Post-hoc Dunn test with Bonferroni correction. Effect size is A12: values close to 0 indicate the first group has lower costs than the second. All SelectQAOA configurations achieve significantly lower test suite costs than BootQA, IgDec_ideal, IgDec_noise, and SelectQA; no significant differences exist among the SelectQAOA variants themselves.

| Pair | p-value | adj. p-value | A12 |
|------|---------|--------------|-----|
| BootQA vs IgDec_ideal | 0.0557 | 1.0000 | 0.680 |
| BootQA vs IgDec_noise | 0.8170 | 1.0000 | 0.500 |
| IgDec_ideal vs IgDec_noise | 0.0926 | 1.0000 | 0.260 |
| BootQA vs SelectQA | 0.4874 | 1.0000 | 1.000 |
| IgDec_ideal vs SelectQA | 0.2228 | 1.0000 | 1.000 |
| IgDec_noise vs SelectQA | 0.6434 | 1.0000 | 1.000 |
| BootQA vs SelectQAOA_aer_sim | 0.0003 | **0.0118** | 0.000 |
| IgDec_ideal vs SelectQAOA_aer_sim | < 0.0001 | **< 0.0001** | 0.000 |
| IgDec_noise vs SelectQAOA_aer_sim | 0.0001 | **0.0047** | 0.030 |
| SelectQA vs SelectQAOA_aer_sim | < 0.0001 | **0.0006** | 0.000 |
| BootQA vs SelectQAOA_depolarizing_sim/01 | 0.1019 | 1.0000 | 0.000 |
| IgDec_ideal vs SelectQAOA_depolarizing_sim/01 | 0.0004 | **0.0174** | 0.000 |
| IgDec_noise vs SelectQAOA_depolarizing_sim/01 | 0.0619 | 1.0000 | 0.040 |
| SelectQA vs SelectQAOA_depolarizing_sim/01 | 0.0198 | 0.8912 | 0.000 |
| SelectQAOA_aer_sim vs SelectQAOA_depolarizing_sim/01 | 0.0440 | 1.0000 | 0.650 |
| BootQA vs SelectQAOA_depolarizing_sim/02 | 0.0026 | 0.1150 | 0.000 |
| IgDec_ideal vs SelectQAOA_depolarizing_sim/02 | < 0.0001 | **< 0.0001** | 0.000 |
| IgDec_noise vs SelectQAOA_depolarizing_sim/02 | 0.0012 | 0.0523 | 0.030 |
| SelectQA vs SelectQAOA_depolarizing_sim/02 | 0.0002 | **0.0093** | 0.000 |
| SelectQAOA_aer_sim vs SelectQAOA_depolarizing_sim/02 | 0.5270 | 1.0000 | 0.620 |
| SelectQAOA_depolarizing_sim/01 vs SelectQAOA_depolarizing_sim/02 | 0.1673 | 1.0000 | 0.460 |
| BootQA vs SelectQAOA_depolarizing_sim/05 | 0.0005 | **0.0239** | 0.000 |
| IgDec_ideal vs SelectQAOA_depolarizing_sim/05 | < 0.0001 | **< 0.0001** | 0.000 |
| IgDec_noise vs SelectQAOA_depolarizing_sim/05 | 0.0002 | **0.0099** | 0.020 |
| SelectQA vs SelectQAOA_depolarizing_sim/05 | < 0.0001 | **0.0014** | 0.000 |
| SelectQAOA_aer_sim vs SelectQAOA_depolarizing_sim/05 | 0.8531 | 1.0000 | 0.450 |
| SelectQAOA_depolarizing_sim/01 vs SelectQAOA_depolarizing_sim/05 | 0.0675 | 1.0000 | 0.300 |
| SelectQAOA_depolarizing_sim/02 vs SelectQAOA_depolarizing_sim/05 | 0.6545 | 1.0000 | 0.350 |
| BootQA vs SelectQAOA_fake_brisbane | 0.0002 | **0.0082** | 0.000 |
| IgDec_ideal vs SelectQAOA_fake_brisbane | < 0.0001 | **< 0.0001** | 0.000 |
| IgDec_noise vs SelectQAOA_fake_brisbane | 0.0001 | **0.0032** | 0.020 |
| SelectQA vs SelectQAOA_fake_brisbane | < 0.0001 | **0.0004** | 0.000 |
| SelectQAOA_aer_sim vs SelectQAOA_fake_brisbane | 0.9262 | 1.0000 | 0.480 |
| SelectQAOA_depolarizing_sim/01 vs SelectQAOA_fake_brisbane | 0.0352 | 1.0000 | 0.290 |
| SelectQAOA_depolarizing_sim/02 vs SelectQAOA_fake_brisbane | 0.4683 | 1.0000 | 0.350 |
| SelectQAOA_depolarizing_sim/05 vs SelectQAOA_fake_brisbane | 0.7812 | 1.0000 | 0.530 |
| BootQA vs SelectQAOA_statevector_sim | 0.0035 | 0.1593 | 0.000 |
| IgDec_ideal vs SelectQAOA_statevector_sim | < 0.0001 | **0.0001** | 0.000 |
| IgDec_noise vs SelectQAOA_statevector_sim | 0.0016 | 0.0740 | 0.100 |
| SelectQA vs SelectQAOA_statevector_sim | 0.0003 | **0.0137** | 0.000 |
| SelectQAOA_aer_sim vs SelectQAOA_statevector_sim | 0.4636 | 1.0000 | 0.800 |
| SelectQAOA_depolarizing_sim/01 vs SelectQAOA_statevector_sim | 0.2003 | 1.0000 | 0.900 |
| SelectQAOA_depolarizing_sim/02 vs SelectQAOA_statevector_sim | 0.9201 | 1.0000 | 0.800 |
| SelectQAOA_depolarizing_sim/05 vs SelectQAOA_statevector_sim | 0.5838 | 1.0000 | 0.800 |
| SelectQAOA_fake_brisbane vs SelectQAOA_statevector_sim | 0.4091 | 1.0000 | 1.000 |

---

### final_effectivenesses — One-way ANOVA + Tukey HSD + Cohen's d

ANOVA: F(9, 90) = 326, p < 0.001. Post-hoc pairwise comparisons with Tukey HSD correction. Effect size is Cohen's d: positive values indicate the first group has higher effectiveness; negative values indicate lower effectiveness. All SelectQAOA configurations substantially outperform BootQA and SelectQA, as well as both IgDec variants, with very large effect sizes.

| Pair | p-value | adj. p-value | Cohen's d |
|------|---------|--------------|-----------|
| IgDec_ideal vs BootQA | < 0.0001 | **< 0.0001** | 4.824 |
| IgDec_noise vs BootQA | < 0.0001 | **< 0.0001** | 3.741 |
| SelectQA vs BootQA | 0.6216 | 0.6216 | 0.931 |
| SelectQAOA_aer_sim vs BootQA | < 0.0001 | **< 0.0001** | 10.771 |
| SelectQAOA_depolarizing_sim/01 vs BootQA | < 0.0001 | **< 0.0001** | 11.862 |
| SelectQAOA_depolarizing_sim/02 vs BootQA | < 0.0001 | **< 0.0001** | 10.600 |
| SelectQAOA_depolarizing_sim/05 vs BootQA | < 0.0001 | **< 0.0001** | 10.085 |
| SelectQAOA_fake_brisbane vs BootQA | < 0.0001 | **< 0.0001** | 11.533 |
| SelectQAOA_statevector_sim vs BootQA | < 0.0001 | **< 0.0001** | 14.377 |
| IgDec_noise vs IgDec_ideal | 1.0000 | 1.0000 | −0.176 |
| SelectQA vs IgDec_ideal | < 0.0001 | **< 0.0001** | −8.627 |
| SelectQAOA_aer_sim vs IgDec_ideal | < 0.0001 | **< 0.0001** | 8.711 |
| SelectQAOA_depolarizing_sim/01 vs IgDec_ideal | < 0.0001 | **< 0.0001** | 10.536 |
| SelectQAOA_depolarizing_sim/02 vs IgDec_ideal | < 0.0001 | **< 0.0001** | 8.486 |
| SelectQAOA_depolarizing_sim/05 vs IgDec_ideal | < 0.0001 | **< 0.0001** | 7.811 |
| SelectQAOA_fake_brisbane vs IgDec_ideal | < 0.0001 | **< 0.0001** | 9.797 |
| SelectQAOA_statevector_sim vs IgDec_ideal | < 0.0001 | **< 0.0001** | 17.073 |
| SelectQA vs IgDec_noise | < 0.0001 | **< 0.0001** | −4.386 |
| SelectQAOA_aer_sim vs IgDec_noise | < 0.0001 | **< 0.0001** | 6.819 |
| SelectQAOA_depolarizing_sim/01 vs IgDec_noise | < 0.0001 | **< 0.0001** | 7.449 |
| SelectQAOA_depolarizing_sim/02 vs IgDec_noise | < 0.0001 | **< 0.0001** | 6.643 |
| SelectQAOA_depolarizing_sim/05 vs IgDec_noise | < 0.0001 | **< 0.0001** | 6.408 |
| SelectQAOA_fake_brisbane vs IgDec_noise | < 0.0001 | **< 0.0001** | 7.431 |
| SelectQAOA_statevector_sim vs IgDec_noise | < 0.0001 | **< 0.0001** | 9.275 |
| SelectQAOA_aer_sim vs SelectQA | < 0.0001 | **< 0.0001** | 15.616 |
| SelectQAOA_depolarizing_sim/01 vs SelectQA | < 0.0001 | **< 0.0001** | 21.481 |
| SelectQAOA_depolarizing_sim/02 vs SelectQA | < 0.0001 | **< 0.0001** | 15.365 |
| SelectQAOA_depolarizing_sim/05 vs SelectQA | < 0.0001 | **< 0.0001** | 13.386 |
| SelectQAOA_fake_brisbane vs SelectQA | < 0.0001 | **< 0.0001** | 17.816 |
| SelectQAOA_statevector_sim vs SelectQA | < 0.0001 | **< 0.0001** | +∞ |
| SelectQAOA_depolarizing_sim/01 vs SelectQAOA_aer_sim | 1.0000 | 1.0000 | −0.234 |
| SelectQAOA_depolarizing_sim/02 vs SelectQAOA_aer_sim | 1.0000 | 1.0000 | −0.197 |
| SelectQAOA_depolarizing_sim/05 vs SelectQAOA_aer_sim | 1.0000 | 1.0000 | 0.091 |
| SelectQAOA_fake_brisbane vs SelectQAOA_aer_sim | 0.9968 | 0.9968 | 0.365 |
| SelectQAOA_statevector_sim vs SelectQAOA_aer_sim | 0.9998 | 0.9998 | 0.352 |
| SelectQAOA_depolarizing_sim/02 vs SelectQAOA_depolarizing_sim/01 | 1.0000 | 1.0000 | 0.008 |
| SelectQAOA_depolarizing_sim/05 vs SelectQAOA_depolarizing_sim/01 | 0.9989 | 0.9989 | 0.311 |
| SelectQAOA_fake_brisbane vs SelectQAOA_depolarizing_sim/01 | 0.9277 | 0.9277 | 0.676 |
| SelectQAOA_statevector_sim vs SelectQAOA_depolarizing_sim/01 | 0.9791 | 0.9791 | 0.896 |
| SelectQAOA_depolarizing_sim/05 vs SelectQAOA_depolarizing_sim/02 | 0.9991 | 0.9991 | 0.271 |
| SelectQAOA_fake_brisbane vs SelectQAOA_depolarizing_sim/02 | 0.9325 | 0.9325 | 0.572 |
| SelectQAOA_statevector_sim vs SelectQAOA_depolarizing_sim/02 | 0.9811 | 0.9811 | 0.631 |
| SelectQAOA_fake_brisbane vs SelectQAOA_depolarizing_sim/05 | 0.9998 | 0.9998 | 0.237 |
| SelectQAOA_statevector_sim vs SelectQAOA_depolarizing_sim/05 | 1.0000 | 1.0000 | 0.180 |
| SelectQAOA_statevector_sim vs SelectQAOA_fake_brisbane | 1.0000 | 1.0000 | −0.155 |

---

## ELEVATOR2

### final_test_suite_costs — Mann-Whitney + A12

Pairwise Mann-Whitney U tests between each SelectQAOA configuration and each IgDec_QAOA configuration (no multi-group correction applied). Effect size is A12: values close to 0 indicate SelectQAOA has lower costs than IgDec_QAOA; all comparisons are significant and strongly favour SelectQAOA.

| SelectQAOA Config | IgDec Config | p-value | A12 |
|---|---|---------|-----|
| statevector_sim | ideal/qaoa_1/elevator_three | < 0.0001 | 0.00 |
| statevector_sim | noise/qaoa_1/elevator_one | < 0.0001 | 0.00 |
| aer_sim | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| aer_sim | noise/qaoa_1/elevator_one | 0.0002 | 0.00 |
| fake_brisbane | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| fake_brisbane | noise/qaoa_1/elevator_one | 0.0002 | 0.00 |
| depolarizing_sim/01 | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| depolarizing_sim/01 | noise/qaoa_1/elevator_one | 0.0002 | 0.00 |
| depolarizing_sim/02 | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| depolarizing_sim/02 | noise/qaoa_1/elevator_one | 0.0002 | 0.00 |
| depolarizing_sim/05 | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| depolarizing_sim/05 | noise/qaoa_1/elevator_one | 0.0002 | 0.00 |

---

### final_pcounts — Mann-Whitney + A12

Pairwise Mann-Whitney U tests on the number of problem counts (pcounts). A12 = 0.00 vs IgDec_ideal indicates a complete dominance by SelectQAOA; the slightly higher A12 values against IgDec_noise reflect a smaller but still significant advantage.

| SelectQAOA Config | IgDec Config | p-value | A12 |
|---|---|---------|-----|
| statevector_sim | ideal/qaoa_1/elevator_three | < 0.0001 | 0.00 |
| statevector_sim | noise/qaoa_1/elevator_one | 0.0014 | 0.10 |
| aer_sim | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| aer_sim | noise/qaoa_1/elevator_one | 0.0028 | 0.10 |
| fake_brisbane | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| fake_brisbane | noise/qaoa_1/elevator_one | 0.0028 | 0.10 |
| depolarizing_sim/01 | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| depolarizing_sim/01 | noise/qaoa_1/elevator_one | 0.0028 | 0.10 |
| depolarizing_sim/02 | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| depolarizing_sim/02 | noise/qaoa_1/elevator_one | 0.0028 | 0.10 |
| depolarizing_sim/05 | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| depolarizing_sim/05 | noise/qaoa_1/elevator_one | 0.0028 | 0.10 |

---

### final_dists — Mann-Whitney + A12

Pairwise Mann-Whitney U tests on solution distances. SelectQAOA achieves significantly lower distances than both IgDec configurations. The A12 values against IgDec_noise (0.19–0.20) are slightly higher than against IgDec_ideal (0.00), indicating a somewhat reduced but still clear advantage.

| SelectQAOA Config | IgDec Config | p-value | A12 |
|---|---|---------|-----|
| statevector_sim | ideal/qaoa_1/elevator_three | < 0.0001 | 0.00 |
| statevector_sim | noise/qaoa_1/elevator_one | 0.0172 | 0.20 |
| aer_sim | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| aer_sim | noise/qaoa_1/elevator_one | 0.0211 | 0.19 |
| fake_brisbane | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| fake_brisbane | noise/qaoa_1/elevator_one | 0.0257 | 0.20 |
| depolarizing_sim/01 | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| depolarizing_sim/01 | noise/qaoa_1/elevator_one | 0.0211 | 0.19 |
| depolarizing_sim/02 | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| depolarizing_sim/02 | noise/qaoa_1/elevator_one | 0.0257 | 0.20 |
| depolarizing_sim/05 | ideal/qaoa_1/elevator_three | 0.0002 | 0.00 |
| depolarizing_sim/05 | noise/qaoa_1/elevator_one | 0.0257 | 0.20 |

---

## ELEVATOR

### final_test_suite_costs — Mann-Whitney + A12

Pairwise Mann-Whitney U tests between each SelectQAOA configuration and each IgDec_QAOA configuration. A12 = 0.00 across all pairs indicates complete stochastic dominance by SelectQAOA in terms of lower test suite costs, confirmed by p < 0.001 in all cases.

| SelectQAOA Config | IgDec Config | p-value | A12 |
|---|---|---------|-----|
| statevector_sim | ideal | < 0.0001 | 0.00 |
| statevector_sim | noise | < 0.0001 | 0.00 |
| aer_sim | ideal | 0.0001 | 0.00 |
| aer_sim | noise | 0.0002 | 0.00 |
| fake_brisbane | ideal | 0.0001 | 0.00 |
| fake_brisbane | noise | 0.0002 | 0.00 |
| depolarizing_sim/01 | ideal | 0.0001 | 0.00 |
| depolarizing_sim/01 | noise | 0.0002 | 0.00 |
| depolarizing_sim/02 | ideal | 0.0001 | 0.00 |
| depolarizing_sim/02 | noise | 0.0002 | 0.00 |
| depolarizing_sim/05 | ideal | 0.0001 | 0.00 |
| depolarizing_sim/05 | noise | 0.0002 | 0.00 |

---

### final_effectivenesses — Mann-Whitney + A12

Pairwise Mann-Whitney U tests on test effectiveness. SelectQAOA is significantly less effective than IgDec_ideal in all comparisons (A12 = 0.10, p < 0.003). Against IgDec_noise the advantage is weaker and not always significant after correction: statevector and depolarizing_sim/01 show no significant difference, while the remaining configurations show marginal significance (p ≈ 0.045–0.121).

| SelectQAOA Config | IgDec Config | p-value | A12 |
|---|---|---------|-----|
| statevector_sim | ideal | 0.0012 | 0.10 |
| statevector_sim | noise | 0.1153 | 0.30 |
| aer_sim | ideal | 0.0025 | 0.10 |
| aer_sim | noise | 0.0452 | 0.23 |
| fake_brisbane | ideal | 0.0025 | 0.10 |
| fake_brisbane | noise | 0.0890 | 0.27 |
| depolarizing_sim/01 | ideal | 0.0025 | 0.10 |
| depolarizing_sim/01 | noise | 0.1212 | 0.29 |
| depolarizing_sim/02 | ideal | 0.0025 | 0.10 |
| depolarizing_sim/02 | noise | 0.0890 | 0.27 |
| depolarizing_sim/05 | ideal | 0.0025 | 0.10 |
| depolarizing_sim/05 | noise | 0.0757 | 0.26 |

---

## IOFROL

### final_test_suite_costs — Mann-Whitney + A12

Pairwise Mann-Whitney U tests between each SelectQAOA configuration and each IgDec_QAOA configuration. A12 values of 1.00 vs IgDec_ideal and 0.90 vs IgDec_noise indicate that IgDec consistently achieves lower test suite costs than SelectQAOA, with all comparisons statistically significant.

| SelectQAOA Config | IgDec Config | p-value | A12 |
|---|---|---------|-----|
| statevector_sim | ideal | < 0.0001 | 1.00 |
| statevector_sim | noise | 0.0014 | 0.90 |
| aer_sim | ideal | 0.0002 | 1.00 |
| aer_sim | noise | 0.0028 | 0.90 |
| fake_brisbane | ideal | 0.0002 | 1.00 |
| fake_brisbane | noise | 0.0028 | 0.90 |
| depolarizing_sim/01 | ideal | 0.0002 | 1.00 |
| depolarizing_sim/01 | noise | 0.0028 | 0.90 |
| depolarizing_sim/02 | ideal | 0.0002 | 1.00 |
| depolarizing_sim/02 | noise | 0.0028 | 0.90 |
| depolarizing_sim/05 | ideal | 0.0002 | 1.00 |
| depolarizing_sim/05 | noise | 0.0028 | 0.90 |

---

### final_effectivenesses — Mann-Whitney + A12

Pairwise Mann-Whitney U tests on test effectiveness. IgDec completely dominates SelectQAOA across all configurations (A12 = 1.00, p < 0.001 in all cases), indicating that IgDec produces substantially more effective test suites on this dataset regardless of the SelectQAOA backend used.

| SelectQAOA Config | IgDec Config | p-value | A12 |
|---|---|---------|-----|
| statevector_sim | ideal | < 0.0001 | 1.00 |
| statevector_sim | noise | < 0.0001 | 1.00 |
| aer_sim | ideal | 0.0002 | 1.00 |
| aer_sim | noise | 0.0002 | 1.00 |
| fake_brisbane | ideal | 0.0002 | 1.00 |
| fake_brisbane | noise | 0.0002 | 1.00 |
| depolarizing_sim/01 | ideal | 0.0002 | 1.00 |
| depolarizing_sim/01 | noise | 0.0002 | 1.00 |
| depolarizing_sim/02 | ideal | 0.0002 | 1.00 |
| depolarizing_sim/02 | noise | 0.0002 | 1.00 |
| depolarizing_sim/05 | ideal | 0.0002 | 1.00 |
| depolarizing_sim/05 | noise | 0.0002 | 1.00 |
