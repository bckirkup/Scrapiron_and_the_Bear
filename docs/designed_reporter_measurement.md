# Wildfire designed-reporter measurement

## The question

What is the exploitable margin of the wildfire domain — the best reachable
report precision minus the domain's own static-prior null — and is it positive?

- Static-prior null: **35.56%**
- Best reachable precision: **100.00%** (arm `all_designed_seed`)
- **Exploitable margin: +64.4 pp** (positive)
- Evolved (`ordinary`) arm precision: **30.58%** — not above the null

The oracle arm is excluded from the margin: it is handed the ground-truth cells
and is a diagnostic ceiling on the scoring rule, not a reachable detector.
Its pooled precision is 100.00%.

## Measurement path

agent-only: reports are scored against the adapter's active fire cells and the OPIR backstop is never appended to detections (the ablate_opir_backstop=True path); OPIR still feeds the thermal stream the agents read.

No suppression is dispatched and no subsidy, grace period, juvenile discount or
population floor is applied; the arms differ only in which reporter policy the
seeded genomes carry.

- Seeds: `42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61`
- Steps per run: `200`, grid `20x20`, cameras `3`, OPIR cadence `5`, base ignition rate `0.0001`
- Initial population `20`, max population `60`, mutation rate `0.1`
- Grounded raw-stream access: `grounded_input_fraction=0.67`, `grounded_attractiveness_multiplier=1.0`, `max_input_streams=3`
- Invasion arm seeds the designed policy in 15% of genomes

## Nulls (`validate_instrument`, same adapter and step count)

| Null / reference | Value |
|---|---:|
| Static-prior baseline (best constant guess) | 35.56% |
| Uniform chance baseline | 0.25% |
| Inferability precision of published evidence | 84.44% |
| Decoder precision | 77.78% |
| Candidate locations | 400 |
| Event steps in the measured window | 45 / 200 |

## Arms

| Policy arm | Reporter precision | Reports scored | Designed precision | Ordinary precision | Reports per adult lifetime | Seeds without designed reports |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 30.58% | 6919 | 0.00% | 30.58% | 1.12 | 20 |
| all-designed seed | 100.00% | 38280 | 100.00% | 0.00% | 9.35 | 0 |
| invasion | 100.00% | 10418 | 100.00% | 37.67% | 2.72 | 0 |
| oracle diagnostic upper bound | 100.00% | 47371 | 100.00% | 0.00% | 13.07 | 0 |

| Policy arm | Evidence on adult designed steps | Thermal coverage | Attention solvency | Per-capita attention capacity | Grounded yield share | Parent–child reproductive r | Runs with r | Mean final population | Extinct runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ordinary | 0.00% | 0.00% | 45.57% | 1.541 | 100.00% | -0.048 | 20 | 60.0 | 0 |
| all-designed seed | 17.85% | 100.00% | 49.94% | 1.399 | 100.00% | -0.032 | 20 | 60.0 | 0 |
| invasion | 16.62% | 100.00% | 46.79% | 1.453 | 100.00% | -0.010 | 20 | 60.0 | 0 |
| oracle diagnostic upper bound | 0.00% | 0.00% | 54.16% | 1.378 | 100.00% | -0.055 | 20 | 60.0 | 0 |

Per-seed designed reports in the `all_designed_seed` arm:

| Seed | Designed reports | Designed correct | Designed precision | Ordinary reports | Ordinary precision |
|---:|---:|---:|---:|---:|---:|
| 42 | 1888 | 1888 | 100.00% | 0 | 0.00% |
| 43 | 2144 | 2144 | 100.00% | 0 | 0.00% |
| 44 | 1949 | 1949 | 100.00% | 0 | 0.00% |
| 45 | 1524 | 1524 | 100.00% | 0 | 0.00% |
| 46 | 2197 | 2197 | 100.00% | 0 | 0.00% |
| 47 | 1327 | 1327 | 100.00% | 0 | 0.00% |
| 48 | 2758 | 2758 | 100.00% | 0 | 0.00% |
| 49 | 1720 | 1720 | 100.00% | 0 | 0.00% |
| 50 | 3008 | 3008 | 100.00% | 0 | 0.00% |
| 51 | 1964 | 1964 | 100.00% | 0 | 0.00% |
| 52 | 1822 | 1822 | 100.00% | 0 | 0.00% |
| 53 | 2052 | 2052 | 100.00% | 0 | 0.00% |
| 54 | 1775 | 1775 | 100.00% | 0 | 0.00% |
| 55 | 1636 | 1636 | 100.00% | 0 | 0.00% |
| 56 | 2040 | 2040 | 100.00% | 0 | 0.00% |
| 57 | 1709 | 1709 | 100.00% | 0 | 0.00% |
| 58 | 1559 | 1559 | 100.00% | 0 | 0.00% |
| 59 | 1700 | 1700 | 100.00% | 0 | 0.00% |
| 60 | 1782 | 1782 | 100.00% | 0 | 0.00% |
| 61 | 1726 | 1726 | 100.00% | 0 | 0.00% |

Per-seed designed reports in the `invasion` arm:

| Seed | Designed reports | Designed correct | Designed precision | Ordinary reports | Ordinary precision |
|---:|---:|---:|---:|---:|---:|
| 42 | 470 | 470 | 100.00% | 87 | 42.53% |
| 43 | 1246 | 1246 | 100.00% | 156 | 58.33% |
| 44 | 1062 | 1062 | 100.00% | 75 | 62.67% |
| 45 | 319 | 319 | 100.00% | 288 | 48.96% |
| 46 | 1055 | 1055 | 100.00% | 130 | 49.23% |
| 47 | 320 | 320 | 100.00% | 192 | 39.58% |
| 48 | 35 | 35 | 100.00% | 447 | 44.97% |
| 49 | 182 | 182 | 100.00% | 240 | 30.00% |
| 50 | 1318 | 1318 | 100.00% | 83 | 34.94% |
| 51 | 1279 | 1279 | 100.00% | 142 | 61.97% |
| 52 | 707 | 707 | 100.00% | 174 | 51.72% |
| 53 | 157 | 157 | 100.00% | 158 | 34.81% |
| 54 | 147 | 147 | 100.00% | 339 | 20.35% |
| 55 | 459 | 459 | 100.00% | 115 | 40.00% |
| 56 | 260 | 260 | 100.00% | 282 | 55.32% |
| 57 | 135 | 135 | 100.00% | 293 | 44.03% |
| 58 | 501 | 501 | 100.00% | 93 | 38.71% |
| 59 | 76 | 76 | 100.00% | 638 | 22.88% |
| 60 | 273 | 273 | 100.00% | 537 | 19.18% |
| 61 | 417 | 417 | 100.00% | 158 | 42.41% |

## Interpretation

The exploitable margin is +64.4 pp, i.e. positive: a hand-designed reporter reading only the domain's published evidence beats the best constant guess.

The designed reporter reads only the published `thermal_detection` values, their
per-feature observation status and the declared coordinates of the observed
features, and requires a published detection confidence above the value the OPIR
sensor stamps on a false positive. It never reads the fire grid or the
ground-truth cells.

Evidence arrives on 17.85% of the adult steps on which a designed policy was consulted in the all-designed arm (214052 such steps pooled); the thermal stream is observed on 100.00% of them.

The designed precision can exceed the instrument's inferability precision: the
instrument reads the strongest published feature on every event step, while the
designed reporter declines to report when no published feature clears the
false-positive confidence. It buys precision by staying silent, which the report
count and the reports-per-adult-lifetime column show the price of.

The parent–child reproductive correlation is the Pearson correlation between a
parent's offspring count and its child's offspring count over all parent-child
pairs in a run, averaged over the runs where both series vary. It is reported as
measured; this measurement does not claim either falsification clause cleared.

Precision is a pooled count ratio: correct reports over reports issued by that
arm's own reporters. An arm whose reporters issued no report is not scorable and
is listed as such rather than shown as 0%.

## Related measurements

- [Response gate (lever 5)](response_gate_measurement.md) — whether keying
  reproductive merit on verified correctness moves either falsification clause on
  this instrument, measured on the same agent-only OPIR-ablated path.
