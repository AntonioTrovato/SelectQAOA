| Dataset | Algorithm | Backend | Execution Cost Mean | Execution Cost Std. Dev. | Failure Rate Mean | Failure Rate Std. Dev. |
|---|---|---:|---:|---:|---:|---:|
| *GSDTSR* | QAOA-TCS | Aer simulator | 14680768.820 | 2205697.681 | **7.867** | 0.105 |
| *GSDTSR* | QAOA-TCS | Fake Brisbane | 14151734.794 | 1476231.618 | 7.858 | 0.079 |
| *GSDTSR* | QAOA-TCS | Depol. 1% | 14339789.085 | 1684971.265 | 7.810 | 0.073 |
| *GSDTSR* | QAOA-TCS | Depol. 2% | 14224328.249 | 1567804.183 | 7.804 | 0.104 |
| *GSDTSR* | QAOA-TCS | Depol. 5% | 14840150.164 | 1495014.503 | 7.823 | 0.072 |
| *GSDTSR* | IGDec-QAOA | Fake Brisbane | 16580166.302 | 9828542.711 | 7.622 | 0.573 |
| *GSDTSR* | SelectQA | -- | **37450.930** | 0.00 | 3.607 | 0.000 |
| *GSDTSR* | BootQA | -- | 177112.778 | 105666.698 | 2.388 | 0.883 |
| *PaintControl* | QAOA-TCS | Aer simulator | 715998.700 | 176124.057 | 10.553 | 1.138 |
| *PaintControl* | QAOA-TCS | Fake Brisbane | 695227.100 | 127316.412 | 10.182 | 1.125 |
| *PaintControl* | QAOA-TCS | Depol. 1% | 715266.900 | 136190.237 | **10.589** | 0.983 |
| *PaintControl* | QAOA-TCS | Depol. 2% | 688505.400 | 101131.551 | 10.264 | 0.571 |
| *PaintControl* | QAOA-TCS | Depol. 5% | 728595.300 | 167952.611 | 10.484 | 1.044 |
| *PaintControl* | IGDec-QAOA | Fake Brisbane | 796327.800 | 167471.318 | 9.780 | 0.697 |
| *PaintControl* | SelectQA | -- | **159150.000** | 0.000 | 5.314 | 0.000 |
| *PaintControl* | BootQA | -- | 651774.600 | 242047.481 | 4.681 | 0.912 |
| *IOF/ROL* | QAOA-TCS | Aer simulator | **44151445.700** | 1217227.391 | 333.166 | 5.328 |
| *IOF/ROL* | QAOA-TCS | Fake Brisbane | 44574781.500 | 865383.586 | 333.200 | 4.514 |
| *IOF/ROL* | QAOA-TCS | Depol. 1% | 44403044.200 | 1235856.388 | **334.201** | 6.879 |
| *IOF/ROL* | QAOA-TCS | Depol. 2% | 44633510.300 | 1242876.992 | 333.284 | 4.980 |
| *IOF/ROL* | QAOA-TCS | Depol. 5% | 44957832.900 | 1365267.961 | 331.080 | 6.304 |
| *IOF/ROL* | IGDec-QAOA | Fake Brisbane | 51540358.300 | 8391230.832 | 334.116 | 21.974 |

*Table: Execution cost and effectiveness (failure rate) for QAOA-TCS with balanced weights under different backends and for the baselines across GSDTSR, PaintControl, and IOF/ROL.*
