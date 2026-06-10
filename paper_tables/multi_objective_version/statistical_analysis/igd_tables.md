# Inverted Generational Distance (IGD) statistical comparison tables

Each table reports pairwise comparisons among the evaluated methods. The caption specifies the simulator, program, metric, and statistical tests used.

**Table 1. IGD results for program FLEX on Statevector (Ideal). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05133 | 0.30798 | 1.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQA | 0.00010 | 0.00058 | 0.0000 |
| Add. Greedy - SelectQAOA | 0.00010 | 0.00058 | 1.0000 |
| DIV-GA - SelectQAOA | 0.05133 | 0.30798 | 1.0000 |
| SelectQA - SelectQAOA | 0.05133 | 0.30798 | 1.0000 |

**Table 2. IGD results for program GREP on Statevector (Ideal). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.00010 | 0.00060 | 1.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQA | 0.05184 | 0.31104 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.05184 | 0.31104 | 1.0000 |
| DIV-GA - SelectQAOA | 0.05184 | 0.31104 | 1.0000 |
| SelectQA - SelectQAOA | 0.00010 | 0.00060 | 1.0000 |

**Table 3. IGD results for program GZIP on Statevector (Ideal). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.08004 | 0.48026 | 1.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQA | 0.00010 | 0.00060 | 0.1000 |
| Add. Greedy - SelectQAOA | 0.00046 | 0.00278 | 1.0000 |
| DIV-GA - SelectQAOA | 0.08004 | 0.48026 | 0.9000 |
| SelectQA - SelectQAOA | 0.03240 | 0.19441 | 1.0000 |

**Table 4. IGD results for program SED on Statevector (Ideal). Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00001 | 0.00004 | 1.8740 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -155.2080 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 7.1029 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -26.0124 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -212.1709 |

**Table 5. IGD results for program FLEX on Aer Sim (Sampling Noise). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05201 | 0.31207 | 1.0000 |
| Add. Greedy - SelectQA | 0.00010 | 0.00061 | 0.0000 |
| DIV-GA - SelectQA | 0.05201 | 0.31207 | 0.0000 |
| Add. Greedy - SelectQAOA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQAOA | 0.00010 | 0.00061 | 1.0000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |

**Table 6. IGD results for program GREP on Aer Sim (Sampling Noise). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.01154 | 0.06923 | 1.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQA | 0.00096 | 0.00574 | 0.3000 |
| Add. Greedy - SelectQAOA | 0.00096 | 0.00574 | 1.0000 |
| DIV-GA - SelectQAOA | 0.43703 | 1.00000 | 1.0000 |
| SelectQA - SelectQAOA | 0.01154 | 0.06923 | 1.0000 |

**Table 7. IGD results for program GZIP on Aer Sim (Sampling Noise). Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | -21.1260 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -23.9991 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 4.8840 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -5.7955 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -11.2180 |

**Table 8. IGD results for program SED on Aer Sim (Sampling Noise). Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 2.2275 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -168.3112 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 7.5253 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -25.5318 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -238.0053 |

**Table 9. IGD results for program FLEX on Fake Brisbane + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05201 | 0.31207 | 1.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQA | 0.00010 | 0.00061 | 0.0000 |
| Add. Greedy - SelectQAOA | 0.00010 | 0.00061 | 1.0000 |
| DIV-GA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |

**Table 10. IGD results for program GREP on Fake Brisbane + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | -10.9656 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -167.7136 |
| SelectQA-DIV-GA | 0.00006 | 0.00038 | 1.6483 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -23.9767 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -123.6621 |

**Table 11. IGD results for program GZIP on Fake Brisbane + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | -22.4517 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -25.4921 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 5.6217 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -5.5921 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -11.7887 |

**Table 12. IGD results for program SED on Fake Brisbane + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00001 | 0.00005 | 1.8802 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -96.6709 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 7.9061 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -24.7666 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -136.7328 |

**Table 13. IGD results for program FLEX on Depolarizing 1% + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05201 | 0.31207 | 1.0000 |
| Add. Greedy - SelectQA | 0.00010 | 0.00061 | 0.0000 |
| DIV-GA - SelectQA | 0.05201 | 0.31207 | 0.0000 |
| Add. Greedy - SelectQAOA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQAOA | 0.00010 | 0.00061 | 1.0000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |

**Table 14. IGD results for program GREP on Depolarizing 1% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | -11.4009 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -351.7508 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 2.5169 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -24.4424 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -264.8758 |

**Table 15. IGD results for program GZIP on Depolarizing 1% + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05201 | 0.31207 | 1.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQA | 0.00010 | 0.00061 | 0.0000 |
| Add. Greedy - SelectQAOA | 0.00010 | 0.00061 | 1.0000 |
| DIV-GA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |

**Table 16. IGD results for program SED on Depolarizing 1% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00027 | 0.00163 | 1.4842 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -131.9429 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 7.9542 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -25.4692 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -182.9448 |

**Table 17. IGD results for program FLEX on Depolarizing 2% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | -16.6452 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -85.0474 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 23.0988 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -11.4667 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -104.1385 |

**Table 18. IGD results for program GREP on Depolarizing 2% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | -10.8414 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -99.3515 |
| SelectQA-DIV-GA | 0.00081 | 0.00486 | 1.4261 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -23.2662 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -73.0403 |

**Table 19. IGD results for program GZIP on Depolarizing 2% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | -19.6699 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -34.0748 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 5.6466 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -6.3812 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -16.9599 |

**Table 20. IGD results for program SED on Depolarizing 2% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 3.7317 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -108.0150 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 5.9429 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -26.1898 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -153.3557 |

**Table 21. IGD results for program FLEX on Depolarizing 5% + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05201 | 0.31207 | 1.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQA | 0.00010 | 0.00061 | 0.0000 |
| Add. Greedy - SelectQAOA | 0.00010 | 0.00061 | 1.0000 |
| DIV-GA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |

**Table 22. IGD results for program GREP on Depolarizing 5% + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05201 | 0.31207 | 1.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| DIV-GA - SelectQA | 0.00010 | 0.00061 | 0.0000 |
| Add. Greedy - SelectQAOA | 0.00010 | 0.00061 | 1.0000 |
| DIV-GA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 1.0000 |

**Table 23. IGD results for program GZIP on Depolarizing 5% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | -22.3355 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -29.6328 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 6.2490 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -5.7308 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -14.0390 |

**Table 24. IGD results for program SED on Depolarizing 5% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00263 | 0.01576 | 1.2559 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | -90.7045 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | 7.9878 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | -24.7785 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | -125.0616 |
