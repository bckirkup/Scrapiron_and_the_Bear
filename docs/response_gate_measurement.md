# Wildfire response-gate measurement (lever 5)

## The question

Does keying reproductive merit on verified correctness — TattleTots'
`reproduction_correctness_weight`, which cleared falsification clause 1 on the
engine's own `SparseSensorScenario` — move either falsification clause on the
wildfire instrument?

Reproduce with:

```bash
uv run --no-sync --no-build python scripts/run_response_gate_experiment.py \
    --steps 600 --weights 0 1 \
    --primary-seeds 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 \
    --holdout-seeds 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120
```

## Measurement path

agent-only: reports are scored against the adapter's active fire cells and the OPIR backstop is never appended to detections (the ablate_opir_backstop=True path); OPIR still feeds the thermal stream the agents read.

The arm under test is the ordinary (evolved) arm. The designed reporter is run at
the same config as a visible ceiling, not as the thing under test; it is a
hand-written evidence-only policy, so its precision says what the instrument
allows, not what evolution found.

## Fixed configuration (identical in every arm)

- Steps per run `600`, grid `20x20`, cameras `3`, OPIR cadence `5`, base ignition rate `0.0001`
- Initial population `20`, max population `60`, mutation rate `0.1`
- Grounded raw-stream access `grounded_input_fraction=0.67`, `max_input_streams=3`
- `correct_report_attention_value=8`, `false_alarm_break_even_precision=0.2`, `reproduction_merit_ordering=True`, `escalation_calibration_in_score_units=True`, starting `escalation_threshold` range `(0.05, 0.3)`
- `reproduction_correctness_weight` is the only quantity that differs between arms
- No subsidy, grace period, juvenile discount or population floor is applied, and no suppression is dispatched, so the fire trajectory at a seed is identical across arms

## Nulls (`validate_instrument`, same adapter and step count)

| Null / reference | Value |
|---|---:|
| Static-prior baseline (best constant guess) | 35.56% |
| Uniform chance baseline | 0.25% |
| Inferability precision of published evidence | 84.44% |
| Candidate locations | 400 |
| Event steps in the measured window | 45 / 600 |

## Seed block `holdout` (seeds 101–120, 20 seeds)

| Quantity | `evolved_w0` | `evolved_w1` | `designed_w1` |
|---|---:|---:|---:|
| Correct-report precision | 6.39% | 6.69% | 100.00% |
| Reports scored | 402354 | 386950 | 45453 |
| Reports per adult lifetime | 27.65 | 29.78 | 16.40 |
| Adults that never report | 4.52% | 4.97% | 48.83% |
| Generations with reports | 22.6 | 14.2 | 4.1 |
| Clause 1: correct-report slope / generation | -0.0061 | -0.0149 | -0.0000 |
| Clause 1: seeds rising | 1 | 1 | 4 |
| Clause 2: parent-child reproductive r | -0.078 | -0.071 | -0.101 |
| Clause 2: seeds above 0.2 | 0 | 0 | 0 |
| Steps where the population cap binds | 86.48% | 80.08% | 97.75% |
| Eligible share of agent-steps | 85.51% | 85.89% | 97.79% |
| Mean offspring, adults with a correct report | 6.107 | 6.465 | 1.640 |
| Mean offspring, adults reporting but never correct | 0.748 | 0.678 | 0.000 |
| Mean offspring, silent adults | 1.032 | 0.997 | 0.655 |
| Parent-child precision r | +0.341 | +0.279 | +0.000 |
| Mean final population | 59.9 | 60.0 | 60.0 |

Per-seed detail, `evolved_w0`:

| Seed | Reports | Correct | Precision | Reports/adult | Clause 1 slope | Clause 2 r | Pairs | Cap binds | Final pop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 21718 | 1400 | 6.45% | 24.08 | -0.0041 | +0.010 | 1148 | 88% | 60 |
| 102 | 22932 | 1542 | 6.72% | 62.28 | -0.0034 | -0.137 | 462 | 96% | 60 |
| 103 | 16445 | 1318 | 8.01% | 27.44 | -0.0097 | -0.044 | 816 | 93% | 59 |
| 104 | 18395 | 857 | 4.66% | 20.23 | -0.0050 | -0.074 | 1200 | 87% | 60 |
| 105 | 20561 | 1378 | 6.70% | 18.63 | -0.0039 | -0.119 | 1463 | 79% | 60 |
| 106 | 17788 | 727 | 4.09% | 14.30 | -0.0013 | -0.088 | 1619 | 73% | 60 |
| 107 | 16015 | 1568 | 9.79% | 22.13 | -0.0066 | -0.136 | 938 | 91% | 60 |
| 108 | 22183 | 1541 | 6.95% | 36.40 | -0.0244 | -0.077 | 808 | 94% | 60 |
| 109 | 21150 | 1503 | 7.11% | 15.72 | -0.0034 | -0.087 | 1780 | 72% | 59 |
| 110 | 20339 | 1181 | 5.81% | 36.67 | +0.0043 | -0.069 | 717 | 88% | 60 |
| 111 | 16454 | 991 | 6.02% | 22.02 | -0.0079 | -0.024 | 969 | 87% | 60 |
| 112 | 18387 | 1270 | 6.91% | 13.87 | -0.0024 | -0.102 | 1773 | 66% | 60 |
| 113 | 19990 | 1684 | 8.42% | 27.27 | -0.0041 | -0.044 | 950 | 88% | 60 |
| 114 | 22661 | 1238 | 5.46% | 19.94 | -0.0024 | -0.082 | 1499 | 79% | 60 |
| 115 | 20075 | 883 | 4.40% | 32.01 | -0.0083 | -0.018 | 812 | 93% | 60 |
| 116 | 14885 | 959 | 6.44% | 21.70 | -0.0063 | -0.059 | 895 | 90% | 60 |
| 117 | 22662 | 1343 | 5.93% | 28.54 | -0.0022 | -0.078 | 1010 | 90% | 60 |
| 118 | 22124 | 1271 | 5.74% | 39.22 | -0.0135 | -0.116 | 706 | 94% | 60 |
| 119 | 22116 | 1509 | 6.82% | 27.76 | -0.0086 | -0.061 | 1027 | 86% | 60 |
| 120 | 25474 | 1565 | 6.14% | 42.80 | -0.0086 | -0.155 | 791 | 95% | 60 |

Per-seed detail, `evolved_w1`:

| Seed | Reports | Correct | Precision | Reports/adult | Clause 1 slope | Clause 2 r | Pairs | Cap binds | Final pop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 24281 | 1664 | 6.85% | 21.00 | -0.0064 | -0.005 | 1503 | 72% | 60 |
| 102 | 19519 | 1420 | 7.27% | 53.59 | -0.0121 | -0.144 | 447 | 94% | 60 |
| 103 | 18455 | 1575 | 8.53% | 35.27 | -0.0315 | +0.010 | 703 | 93% | 60 |
| 104 | 22389 | 1110 | 4.96% | 25.92 | -0.0108 | -0.002 | 1213 | 79% | 60 |
| 105 | 17959 | 1310 | 7.29% | 9.91 | -0.0026 | -0.008 | 2425 | 41% | 60 |
| 106 | 13151 | 863 | 6.56% | 14.53 | -0.0188 | -0.045 | 1163 | 78% | 60 |
| 107 | 23785 | 1484 | 6.24% | 64.20 | -0.0179 | -0.107 | 481 | 94% | 60 |
| 108 | 24118 | 1923 | 7.97% | 25.10 | -0.0131 | -0.050 | 1305 | 80% | 60 |
| 109 | 18372 | 1428 | 7.77% | 17.51 | -0.0186 | -0.073 | 1362 | 75% | 60 |
| 110 | 11585 | 792 | 6.84% | 24.51 | +0.0033 | -0.149 | 598 | 87% | 60 |
| 111 | 13550 | 937 | 6.92% | 19.16 | -0.0318 | -0.063 | 905 | 86% | 60 |
| 112 | 19230 | 1213 | 6.31% | 18.70 | -0.0092 | -0.059 | 1347 | 71% | 60 |
| 113 | 19376 | 1622 | 8.37% | 21.91 | -0.0301 | -0.042 | 1149 | 83% | 60 |
| 114 | 21023 | 878 | 4.18% | 17.43 | -0.0072 | -0.069 | 1593 | 57% | 60 |
| 115 | 18637 | 1006 | 5.40% | 21.84 | -0.0110 | -0.032 | 1086 | 71% | 60 |
| 116 | 15040 | 893 | 5.94% | 18.08 | -0.0146 | -0.069 | 1120 | 77% | 60 |
| 117 | 19748 | 1280 | 6.48% | 26.29 | -0.0197 | -0.061 | 955 | 89% | 60 |
| 118 | 18673 | 1235 | 6.61% | 56.37 | -0.0398 | -0.063 | 439 | 95% | 60 |
| 119 | 26430 | 1679 | 6.35% | 35.65 | -0.0047 | -0.203 | 989 | 86% | 60 |
| 120 | 21629 | 1588 | 7.34% | 68.62 | -0.0013 | -0.180 | 393 | 95% | 60 |

Per-seed detail, `designed_w1`:

| Seed | Reports | Correct | Precision | Reports/adult | Clause 1 slope | Clause 2 r | Pairs | Cap binds | Final pop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 2251 | 2251 | 100.00% | 16.66 | +0.0000 | -0.126 | 154 | 99% | 60 |
| 102 | 2569 | 2569 | 100.00% | 19.17 | +0.0000 | -0.134 | 159 | 97% | 60 |
| 103 | 2420 | 2420 | 100.00% | 20.78 | -0.0000 | +0.005 | 129 | 99% | 60 |
| 104 | 1807 | 1807 | 100.00% | 11.51 | -0.0000 | -0.261 | 187 | 97% | 60 |
| 105 | 2405 | 2405 | 100.00% | 17.06 | -0.0000 | -0.180 | 159 | 98% | 60 |
| 106 | 1740 | 1740 | 100.00% | 11.08 | -0.0000 | +0.042 | 184 | 98% | 60 |
| 107 | 2585 | 2585 | 100.00% | 17.01 | -0.0000 | -0.155 | 206 | 98% | 60 |
| 108 | 2702 | 2702 | 100.00% | 25.02 | -0.0000 | -0.180 | 117 | 98% | 60 |
| 109 | 2604 | 2604 | 100.00% | 17.84 | -0.0000 | -0.129 | 168 | 98% | 60 |
| 110 | 2580 | 2580 | 100.00% | 19.69 | -0.0000 | +0.085 | 149 | 97% | 60 |
| 111 | 1832 | 1832 | 100.00% | 10.71 | +0.0000 | -0.111 | 201 | 98% | 60 |
| 112 | 2376 | 2376 | 100.00% | 14.31 | -0.0000 | -0.151 | 205 | 97% | 60 |
| 113 | 2326 | 2326 | 100.00% | 13.84 | -0.0000 | -0.099 | 195 | 98% | 60 |
| 114 | 1978 | 1978 | 100.00% | 17.50 | -0.0000 | -0.110 | 125 | 98% | 60 |
| 115 | 1436 | 1436 | 100.00% | 10.95 | -0.0000 | -0.140 | 152 | 98% | 60 |
| 116 | 2130 | 2130 | 100.00% | 16.77 | -0.0000 | -0.131 | 140 | 97% | 60 |
| 117 | 2175 | 2175 | 100.00% | 11.07 | +0.0000 | -0.061 | 233 | 98% | 60 |
| 118 | 2396 | 2396 | 100.00% | 15.26 | -0.0000 | -0.070 | 194 | 97% | 60 |
| 119 | 2790 | 2790 | 100.00% | 22.68 | -0.0000 | -0.138 | 134 | 96% | 60 |
| 120 | 2351 | 2351 | 100.00% | 19.11 | -0.0000 | +0.025 | 149 | 98% | 60 |

## Seed block `primary` (seeds 42–61, 20 seeds)

| Quantity | `evolved_w0` | `evolved_w1` | `designed_w1` |
|---|---:|---:|---:|
| Correct-report precision | 7.34% | 7.52% | 100.00% |
| Reports scored | 360926 | 336437 | 43464 |
| Reports per adult lifetime | 22.54 | 21.13 | 15.13 |
| Adults that never report | 6.20% | 6.19% | 47.81% |
| Generations with reports | 26.9 | 18.7 | 4.3 |
| Clause 1: correct-report slope / generation | -0.0054 | -0.0107 | -0.0000 |
| Clause 1: seeds rising | 1 | 1 | 1 |
| Clause 2: parent-child reproductive r | -0.080 | -0.064 | -0.114 |
| Clause 2: seeds above 0.2 | 0 | 0 | 0 |
| Steps where the population cap binds | 84.23% | 73.18% | 97.43% |
| Eligible share of agent-steps | 85.06% | 83.63% | 97.79% |
| Mean offspring, adults with a correct report | 6.077 | 6.701 | 1.571 |
| Mean offspring, adults reporting but never correct | 0.805 | 0.779 | 0.000 |
| Mean offspring, silent adults | 1.101 | 1.134 | 0.706 |
| Parent-child precision r | +0.357 | +0.293 | +0.000 |
| Mean final population | 60.0 | 59.6 | 60.0 |

Per-seed detail, `evolved_w0`:

| Seed | Reports | Correct | Precision | Reports/adult | Clause 1 slope | Clause 2 r | Pairs | Cap binds | Final pop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 15842 | 1008 | 6.36% | 23.15 | -0.0106 | -0.064 | 911 | 90% | 60 |
| 43 | 22316 | 1354 | 6.07% | 24.35 | -0.0054 | -0.059 | 1244 | 85% | 60 |
| 44 | 17520 | 1632 | 9.32% | 14.09 | -0.0035 | -0.026 | 1634 | 73% | 60 |
| 45 | 17744 | 1220 | 6.88% | 15.94 | -0.0019 | -0.012 | 1463 | 75% | 60 |
| 46 | 19191 | 1442 | 7.51% | 24.81 | -0.0082 | -0.054 | 997 | 90% | 60 |
| 47 | 14911 | 802 | 5.38% | 26.11 | -0.0097 | -0.105 | 746 | 92% | 60 |
| 48 | 23260 | 2387 | 10.26% | 19.94 | -0.0031 | -0.079 | 1510 | 80% | 60 |
| 49 | 15186 | 969 | 6.38% | 29.91 | -0.0096 | -0.146 | 660 | 93% | 60 |
| 50 | 18275 | 1622 | 8.88% | 55.20 | +0.0000 | -0.105 | 409 | 93% | 60 |
| 51 | 20350 | 1240 | 6.09% | 20.46 | -0.0034 | -0.016 | 1290 | 86% | 60 |
| 52 | 20443 | 1511 | 7.39% | 37.57 | -0.0132 | -0.127 | 709 | 94% | 60 |
| 53 | 15118 | 1108 | 7.33% | 19.14 | -0.0040 | -0.114 | 1019 | 89% | 60 |
| 54 | 16288 | 1000 | 6.14% | 15.68 | -0.0016 | -0.051 | 1432 | 80% | 60 |
| 55 | 17273 | 1088 | 6.30% | 13.30 | -0.0023 | -0.063 | 1728 | 72% | 60 |
| 56 | 17645 | 1544 | 8.75% | 18.74 | -0.0034 | -0.110 | 1223 | 79% | 60 |
| 57 | 18974 | 1574 | 8.30% | 17.01 | -0.0035 | -0.113 | 1497 | 82% | 60 |
| 58 | 16487 | 1181 | 7.16% | 17.40 | -0.0041 | -0.104 | 1230 | 85% | 60 |
| 59 | 21074 | 1473 | 6.99% | 14.98 | -0.0012 | -0.082 | 1914 | 72% | 60 |
| 60 | 20162 | 1419 | 7.04% | 29.49 | -0.0132 | -0.063 | 867 | 94% | 60 |
| 61 | 12867 | 903 | 7.02% | 13.49 | -0.0064 | -0.103 | 1232 | 79% | 60 |

Per-seed detail, `evolved_w1`:

| Seed | Reports | Correct | Precision | Reports/adult | Clause 1 slope | Clause 2 r | Pairs | Cap binds | Final pop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 17329 | 1101 | 6.35% | 49.35 | -0.0087 | -0.141 | 440 | 95% | 60 |
| 43 | 12057 | 1146 | 9.50% | 11.42 | -0.0071 | -0.064 | 1394 | 68% | 59 |
| 44 | 15010 | 1144 | 7.62% | 13.25 | -0.0125 | -0.043 | 1467 | 67% | 60 |
| 45 | 17644 | 1211 | 6.86% | 12.83 | -0.0058 | -0.049 | 1773 | 56% | 60 |
| 46 | 18285 | 1789 | 9.78% | 21.69 | -0.0128 | -0.066 | 1113 | 80% | 60 |
| 47 | 15404 | 1002 | 6.50% | 20.48 | -0.0129 | -0.111 | 994 | 84% | 60 |
| 48 | 24662 | 2366 | 9.59% | 20.79 | -0.0103 | -0.039 | 1533 | 68% | 60 |
| 49 | 19430 | 920 | 4.73% | 53.79 | -0.0278 | -0.110 | 446 | 95% | 60 |
| 50 | 18347 | 1660 | 9.05% | 57.87 | +0.0002 | -0.093 | 394 | 93% | 60 |
| 51 | 13866 | 1069 | 7.71% | 9.72 | -0.0033 | -0.045 | 1900 | 56% | 60 |
| 52 | 18646 | 1274 | 6.83% | 12.64 | -0.0067 | -0.032 | 1911 | 59% | 58 |
| 53 | 22481 | 1337 | 5.95% | 26.78 | -0.0143 | -0.044 | 1097 | 84% | 60 |
| 54 | 15900 | 1137 | 7.15% | 16.69 | -0.0119 | -0.031 | 1255 | 81% | 60 |
| 55 | 12330 | 854 | 6.93% | 11.05 | -0.0061 | -0.149 | 1489 | 67% | 60 |
| 56 | 19512 | 1540 | 7.89% | 13.40 | -0.0055 | -0.048 | 1921 | 66% | 58 |
| 57 | 12205 | 1218 | 9.98% | 11.61 | -0.0128 | -0.051 | 1458 | 67% | 60 |
| 58 | 11626 | 951 | 8.18% | 12.08 | -0.0079 | -0.048 | 1297 | 71% | 60 |
| 59 | 19012 | 1353 | 7.12% | 10.97 | -0.0027 | +0.019 | 2332 | 46% | 58 |
| 60 | 20508 | 1503 | 7.33% | 22.51 | -0.0211 | -0.106 | 1165 | 83% | 60 |
| 61 | 12183 | 719 | 5.90% | 13.60 | -0.0249 | -0.020 | 1169 | 76% | 60 |

Per-seed detail, `designed_w1`:

| Seed | Reports | Correct | Precision | Reports/adult | Clause 1 slope | Clause 2 r | Pairs | Cap binds | Final pop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2031 | 2031 | 100.00% | 16.93 | -0.0000 | -0.023 | 134 | 97% | 60 |
| 43 | 2319 | 2319 | 100.00% | 16.33 | +0.0000 | -0.139 | 162 | 96% | 60 |
| 44 | 2280 | 2280 | 100.00% | 9.34 | -0.0000 | -0.119 | 285 | 97% | 60 |
| 45 | 1846 | 1846 | 100.00% | 16.11 | -0.0000 | -0.165 | 120 | 98% | 60 |
| 46 | 2403 | 2403 | 100.00% | 12.65 | -0.0000 | -0.180 | 223 | 97% | 60 |
| 47 | 1683 | 1683 | 100.00% | 9.90 | -0.0000 | -0.184 | 210 | 97% | 60 |
| 48 | 3184 | 3184 | 100.00% | 26.95 | -0.0000 | +0.097 | 132 | 98% | 60 |
| 49 | 1917 | 1917 | 100.00% | 13.99 | -0.0000 | -0.158 | 155 | 98% | 60 |
| 50 | 3052 | 3052 | 100.00% | 21.49 | -0.0000 | +0.026 | 152 | 96% | 60 |
| 51 | 2229 | 2229 | 100.00% | 12.18 | -0.0000 | -0.163 | 214 | 97% | 60 |
| 52 | 2096 | 2096 | 100.00% | 19.19 | -0.0000 | -0.239 | 121 | 98% | 60 |
| 53 | 2283 | 2283 | 100.00% | 15.64 | -0.0000 | -0.069 | 172 | 97% | 60 |
| 54 | 2158 | 2158 | 100.00% | 13.40 | -0.0000 | -0.160 | 210 | 98% | 60 |
| 55 | 1846 | 1846 | 100.00% | 12.82 | -0.0000 | -0.087 | 171 | 97% | 60 |
| 56 | 2219 | 2219 | 100.00% | 14.41 | -0.0000 | -0.134 | 190 | 96% | 60 |
| 57 | 2129 | 2129 | 100.00% | 15.54 | -0.0000 | -0.181 | 164 | 98% | 60 |
| 58 | 1765 | 1765 | 100.00% | 11.24 | -0.0000 | -0.122 | 188 | 97% | 60 |
| 59 | 1928 | 1928 | 100.00% | 13.11 | -0.0000 | -0.108 | 173 | 98% | 60 |
| 60 | 2113 | 2113 | 100.00% | 16.01 | -0.0000 | -0.061 | 159 | 98% | 60 |
| 61 | 1983 | 1983 | 100.00% | 15.33 | -0.0000 | -0.102 | 145 | 98% | 60 |

## Verdict against the falsification test

A clause is judged met here only if the evolved arm's realized correct-report rate is above this domain's static-prior null (35.56%) and the clause's own criterion holds in at least 80% of seeds pooled over every seed block.

### `evolved_w1`

- **Clause 1** (correct-report rate rises over generations within a run at fixed initial parameters): rising in 2/40 seeds; realized precision not above the static-prior null. **not met**.
- **Clause 2** (parent-child reproductive correlation reliably above 0.2): cleared in 0/40 seeds. **not met**.


## Reading the numbers: measured effect versus artifact

- **Precision here is far below the committed 30.58% no-lever figure, and that is a
  consequence of levers 1-4, not of lever 5.** Paying attention income for verified
  correct reports makes reporting profitable in expectation, so evolved agents report
  around 20-30 times per adult lifetime instead of staying mostly silent, and only
  ~5% of adults never report. The no-lever arm buys its higher precision with silence.
  Report volume is printed next to precision for exactly this reason.
- **The cap-binding share is not the limit on clause 2 here.** On `SparseSensorScenario`
  the cap bound on roughly a third of steps, which was the stated reason ordering
  reproduction could only shift ~0.03 offspring. On this instrument the cap binds on
  73-86% of steps at a stable population of 60, and clause 2 still does not move: the
  parent-child reproductive correlation stays slightly negative in both arms. A binding
  cap is therefore not sufficient for lineage-level heritability of reproductive output.
- **The large offspring gap between adults with a correct report and adults that never
  reported correctly is a lifetime confound, not a gate effect.** It is the same size in
  the `W=0` reserves-only control as at `W=1`; long-lived agents accumulate both more
  correct reports and more offspring. The same applies to the positive parent-child
  *precision* correlation, which is present in the control as well.
- **Clause-1 slopes are negative in both arms and about twice as negative at `W=1`.**
  Turning the gate on makes the within-run trend worse, not better, at comparable report
  volume: the rate declines over generations in 38 of 40 seeds in both arms. That is
  the opposite sign to the SparseSensor result, so lever 5 does not transfer to this
  domain rather than merely falling short of a threshold.
- The designed ceiling at the same config still scores 100.00% precision, so the
  exploitable margin remains large and the negative evolved result is not a thin-domain
  artifact.
