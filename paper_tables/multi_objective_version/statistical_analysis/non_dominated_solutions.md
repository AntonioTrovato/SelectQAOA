### Table: Dunn’s test with Bonferroni correction and $\hat{A}_{12}$ effect size of non-dominated solutions for QAOA-TCS executed on the Ideal Simulator compared to state-of-the-art methods.

| Program | Hypothesis             | $p$-value | Adj. $p$-value | $\hat{A}_{12}$ |
| ------- | ---------------------- | --------: | -------------: | -------------: |
| Flex    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Flex    | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Flex    | QAOA-TCS > Add. Greedy |     0.050 |            0.3 |             -- |
| Grep    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > Add. Greedy |     0.051 |            0.3 |             -- |
| Gzip    | DIV-GA > QAOA-TCS      |     0.051 |            0.3 |             -- |
| Gzip    | DIV-GA > SelectQA      |    < 0.01 |         < 0.01 |          1 (L) |
| Gzip    | DIV-GA > Add. Greedy   |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > DIV-GA      |    < 0.01 |           0.08 |          1 (L) |
| Sed     | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |

### Table: Dunn’s test with Bonferroni correction and $\hat{A}_{12}$ effect size on a simulator with sampling noise.

| Program | Hypothesis             | $p$-value | Adj. $p$-value | $\hat{A}_{12}$ |
| ------- | ---------------------- | --------: | -------------: | -------------: |
| Flex    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Flex    | QAOA-TCS > SelectQA    |     0.051 |            0.3 |             -- |
| Flex    | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > Add. Greedy |     0.051 |           0.30 |             -- |
| Gzip    | DIV-GA > QAOA-TCS      |       0.6 |              1 |             -- |
| Gzip    | DIV-GA > SelectQA      |    < 0.01 |         < 0.01 |          1 (L) |
| Gzip    | DIV-GA > Add. Greedy   |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > DIV-GA      |    < 0.05 |           0.11 |          1 (L) |
| Sed     | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |

### Table: Dunn’s test with Bonferroni correction and $\hat{A}_{12}$ effect size (Fake Brisbane).

| Program | Hypothesis             | $p$-value | Adj. $p$-value | $\hat{A}_{12}$ |
| ------- | ---------------------- | --------: | -------------: | -------------: |
| Flex    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Flex    | QAOA-TCS > SelectQA    |     0.051 |           0.31 |             -- |
| Flex    | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > Add. Greedy |     0.051 |           0.30 |             -- |
| Gzip    | DIV-GA > QAOA-TCS      |     0.055 |           0.33 |       0.95 (L) |
| Gzip    | DIV-GA > SelectQA      |    < 0.01 |         < 0.01 |          1 (L) |
| Gzip    | DIV-GA > Add. Greedy   |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > DIV-GA      |    < 0.05 |         < 0.05 |          1 (L) |
| Sed     | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |

### Table: Dunn’s test with Bonferroni correction and $\hat{A}_{12}$ effect size (1% Depolarizing Error).

| Program | Hypothesis             | $p$-value | Adj. $p$-value | $\hat{A}_{12}$ |
| ------- | ---------------------- | --------: | -------------: | -------------: |
| Flex    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Flex    | QAOA-TCS > SelectQA    |     0.051 |           0.31 |             -- |
| Flex    | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > Add. Greedy |     0.052 |           0.31 |             -- |
| Gzip    | DIV-GA > QAOA-TCS      |      0.10 |           0.62 |             -- |
| Gzip    | DIV-GA > SelectQA      |    < 0.01 |         < 0.01 |          1 (L) |
| Gzip    | DIV-GA > Add. Greedy   |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > DIV-GA      |    < 0.05 |           0.19 |          1 (L) |
| Sed     | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |

### Table: Dunn’s test with Bonferroni correction and $\hat{A}_{12}$ effect size (2% Depolarizing Error).

| Program | Hypothesis             | $p$-value | Adj. $p$-value | $\hat{A}_{12}$ |
| ------- | ---------------------- | --------: | -------------: | -------------: |
| Flex    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Flex    | QAOA-TCS > SelectQA    |     0.051 |           0.31 |          1 (L) |
| Flex    | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > Add. Greedy |     0.051 |           0.30 |             -- |
| Gzip    | DIV-GA > QAOA-TCS      |      0.12 |           0.73 |             -- |
| Gzip    | DIV-GA > SelectQA      |    < 0.01 |         < 0.01 |          1 (L) |
| Gzip    | DIV-GA > Add. Greedy   |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > DIV-GA      |    < 0.05 |           0.06 |          1 (L) |
| Sed     | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |

### Table: Dunn’s test with Bonferroni correction and $\hat{A}_{12}$ effect size (5% Depolarizing Error).

| Program | Hypothesis             | $p$-value | Adj. $p$-value | $\hat{A}_{12}$ |
| ------- | ---------------------- | --------: | -------------: | -------------: |
| Flex    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Flex    | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |             -- |
| Flex    | QAOA-TCS > Add. Greedy |     0.053 |           0.31 |          1 (L) |
| Grep    | QAOA-TCS > DIV-GA      |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Grep    | QAOA-TCS > Add. Greedy |     0.053 |           0.31 |          1 (L) |
| Gzip    | DIV-GA > QAOA-TCS      |     0.053 |           0.31 |             -- |
| Gzip    | DIV-GA > SelectQA      |    < 0.01 |         < 0.01 |          1 (L) |
| Gzip    | DIV-GA > Add. Greedy   |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > DIV-GA      |    < 0.05 |           0.19 |          1 (L) |
| Sed     | QAOA-TCS > SelectQA    |    < 0.01 |         < 0.01 |          1 (L) |
| Sed     | QAOA-TCS > Add. Greedy |    < 0.01 |         < 0.01 |          1 (L) |
