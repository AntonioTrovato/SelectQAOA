# Hypervolume (HV) statistical comparison tables

Each table reports pairwise comparisons among the evaluated methods. The caption specifies the simulator, program, metric, and statistical tests used.

**Table 1. HV results for program FLEX on Statevector (Ideal). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05133 | 0.30798 | 0.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 0.0000 |
| DIV-GA - SelectQA | 0.00010 | 0.00058 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.00010 | 0.00058 | 0.0000 |
| DIV-GA - SelectQAOA | 0.05133 | 0.30798 | 0.0000 |
| SelectQA - SelectQAOA | 0.05133 | 0.30798 | 0.0000 |

**Table 2. HV results for program GREP on Statevector (Ideal). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05184 | 0.31104 | 0.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 0.0000 |
| DIV-GA - SelectQA | 0.00010 | 0.00060 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.00010 | 0.00060 | 0.0000 |
| DIV-GA - SelectQAOA | 0.05184 | 0.31104 | 0.0000 |
| SelectQA - SelectQAOA | 0.05184 | 0.31104 | 0.0000 |

**Table 3. HV results for program GZIP on Statevector (Ideal). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05178 | 0.31070 | 0.0000 |
| Add. Greedy - SelectQA | 0.00010 | 0.00060 | 0.0000 |
| DIV-GA - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.05178 | 0.31070 | 0.0000 |
| DIV-GA - SelectQAOA | 0.00010 | 0.00060 | 1.0000 |
| SelectQA - SelectQAOA | 0.05178 | 0.31070 | 0.0000 |

**Table 4. HV results for program SED on Statevector (Ideal). Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 11.0030 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 62.7185 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -19.8666 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 13.0666 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 84.8882 |

**Table 5. HV results for program FLEX on Aer Sim (Sampling Noise). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.43703 | 1.00000 | 0.0000 |
| Add. Greedy - SelectQA | 0.00001 | 0.00005 | 0.0000 |
| DIV-GA - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.01154 | 0.06923 | 0.0000 |
| DIV-GA - SelectQAOA | 0.00096 | 0.00574 | 0.7000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 0.0000 |

**Table 6. HV results for program GREP on Aer Sim (Sampling Noise). Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 13.7965 |
| SelectQA-Add. Greedy | 0.95127 | 1.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 52.1380 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -13.5925 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 18.4005 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 51.8442 |

**Table 7. HV results for program GZIP on Aer Sim (Sampling Noise). Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 15.1767 |
| SelectQA-Add. Greedy | 0.00003 | 0.00015 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 6.7009 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -9.4785 |
| SelectQAOA-DIV-GA | 0.00001 | 0.00004 | 1.8484 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 4.9124 |

**Table 8. HV results for program SED on Aer Sim (Sampling Noise). Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05201 | 0.31207 | 0.0000 |
| Add. Greedy - SelectQA | 0.00010 | 0.00061 | 1.0000 |
| DIV-GA - SelectQA | 0.05201 | 0.31207 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.00000 | 0.00000 | 0.0000 |
| DIV-GA - SelectQAOA | 0.00010 | 0.00061 | 0.0000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 0.0000 |

**Table 9. HV results for program FLEX on Fake Brisbane + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.69757 | 1.00000 | 0.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 0.0000 |
| DIV-GA - SelectQA | 0.00000 | 0.00002 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.00188 | 0.01127 | 0.0000 |
| DIV-GA - SelectQAOA | 0.00652 | 0.03914 | 0.4000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 0.0000 |

**Table 10. HV results for program GREP on Fake Brisbane + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 13.7965 |
| SelectQA-Add. Greedy | 0.92842 | 1.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 105.2751 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -13.5925 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 21.4244 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 104.6863 |

**Table 11. HV results for program GZIP on Fake Brisbane + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 15.1767 |
| SelectQA-Add. Greedy | 0.00000 | 0.00001 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 7.7629 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -9.4785 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00001 | 1.9717 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 5.6384 |

**Table 12. HV results for program SED on Fake Brisbane + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.05201 | 0.31207 | 0.0000 |
| Add. Greedy - SelectQA | 0.00010 | 0.00061 | 1.0000 |
| DIV-GA - SelectQA | 0.05201 | 0.31207 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.00000 | 0.00000 | 0.0000 |
| DIV-GA - SelectQAOA | 0.00010 | 0.00061 | 0.0000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 0.0000 |

**Table 13. HV results for program FLEX on Depolarizing 1% + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.69757 | 1.00000 | 0.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00002 | 0.0000 |
| DIV-GA - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.00652 | 0.03914 | 0.0000 |
| DIV-GA - SelectQAOA | 0.00188 | 0.01127 | 0.6000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 0.0000 |

**Table 14. HV results for program GREP on Depolarizing 1% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 13.7965 |
| SelectQA-Add. Greedy | 0.92596 | 1.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 120.8566 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -13.5925 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 21.9555 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 120.1857 |

**Table 15. HV results for program GZIP on Depolarizing 1% + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.09472 | 0.56831 | 0.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 0.0000 |
| DIV-GA - SelectQA | 0.00006 | 0.00035 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.00018 | 0.00106 | 0.0000 |
| DIV-GA - SelectQAOA | 0.03761 | 0.22567 | 0.0700 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 0.0000 |

**Table 16. HV results for program SED on Depolarizing 1% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 11.0149 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 38.6746 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -19.8849 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 11.5394 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 52.5582 |

**Table 17. HV results for program FLEX on Depolarizing 2% + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.69757 | 1.00000 | 0.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00002 | 0.0000 |
| DIV-GA - SelectQA | 0.00000 | 0.00000 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.00652 | 0.03914 | 0.0000 |
| DIV-GA - SelectQAOA | 0.00188 | 0.01127 | 0.6000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 0.0000 |

**Table 18. HV results for program GREP on Depolarizing 2% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 13.7965 |
| SelectQA-Add. Greedy | 0.92275 | 1.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 149.7877 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -13.5925 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 21.8485 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 148.9455 |

**Table 19. HV results for program GZIP on Depolarizing 2% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 15.1767 |
| SelectQA-Add. Greedy | 0.00001 | 0.00004 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 6.6459 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -9.4785 |
| SelectQAOA-DIV-GA | 0.00053 | 0.00318 | 1.3885 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 4.7015 |

**Table 20. HV results for program SED on Depolarizing 2% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 11.0030 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 44.7281 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -19.8666 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 11.5557 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 61.1556 |

**Table 21. HV results for program FLEX on Depolarizing 5% + Sampling Noise. Statistical tests used: Kruskal-Wallis + Dunn test with Bonferroni correction + A12 effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | A12 |
|---|---:|---:|---:|
| Add. Greedy - DIV-GA | 0.43703 | 1.00000 | 0.0000 |
| Add. Greedy - SelectQA | 0.00000 | 0.00000 | 0.0000 |
| DIV-GA - SelectQA | 0.00001 | 0.00005 | 1.0000 |
| Add. Greedy - SelectQAOA | 0.00096 | 0.00574 | 0.0000 |
| DIV-GA - SelectQAOA | 0.01154 | 0.06923 | 0.3000 |
| SelectQA - SelectQAOA | 0.05201 | 0.31207 | 0.0000 |

**Table 22. HV results for program GREP on Depolarizing 5% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 13.7965 |
| SelectQA-Add. Greedy | 0.92581 | 1.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 122.0023 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -13.5925 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 21.9962 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 121.3256 |

**Table 23. HV results for program GZIP on Depolarizing 5% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 15.2143 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 8.7427 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -9.4831 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 2.1447 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 6.3261 |

**Table 24. HV results for program SED on Depolarizing 5% + Sampling Noise. Statistical tests used: ANOVA + Tukey HSD with Bonferroni correction + Cohen's d effect size.**

| Comparison | p-value | Adj. p-value (Bonferroni) | Cohen's d |
|---|---:|---:|---:|
| DIV-GA-Add. Greedy | 0.00000 | 0.00000 | 11.0077 |
| SelectQA-Add. Greedy | 0.00000 | 0.00000 | NA |
| SelectQAOA-Add. Greedy | 0.00000 | 0.00000 | 29.7895 |
| SelectQA-DIV-GA | 0.00000 | 0.00000 | -19.8694 |
| SelectQAOA-DIV-GA | 0.00000 | 0.00000 | 10.7100 |
| SelectQAOA-SelectQA | 0.00000 | 0.00000 | 40.3547 |
