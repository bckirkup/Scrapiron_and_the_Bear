# Fire sensor-driven stream re-baseline

## Measurement provenance

These are real measurements from two engine configurations, not a
reconstruction of one result. The Fire source for both measurements is
`42eb583` (`devin/1786749308-refresh-tattletots-pin`). The prior-engine
measurement used TattleTots `d4cfee1`; the current-engine measurement uses
TattleTots `cee59f93f6973fa7fefb2f87dbb40a8ce0095113`. Both comparison and
instrument measurements used the committed harness:

```text
uv run --frozen python scripts/rebaseline_fire_measurements.py
```

The harness runs the 100-step comparison and OPIR-ablation arms plus the
200-step instrument validation. The comparison configuration was seed 42, a
20x20 grid, three cameras, and OPIR cadence 5. Instrument validation used the
same seed, grid, cameras, and cadence, with 400 candidate locations and 45
event steps. The thermal-density measurement was not produced by this
harness: it came from an ad-hoc script that was never committed and no longer
exists. The current-engine comparison values are preserved in the
user-supplied raw measurement output.

Engine-coupled arms, specifically A4 and its OPIR ablation, move when the
TattleTots engine pin moves. A0-A3 did not move, and instrument validation did
not move.

## Why the data path changed

The previous `thermal_detection` stream copied true fire intensity from every
sampled grid cell. It did not call `OPIRSatellite.scan` or
`CameraTower.detect`, even though the conventional architectures use those
sensor models. Consequently, TattleTots received noiseless ground truth while
the comparison baselines received OPIR misses, false positives, intensity
thresholding, camera range/LOS limits, and night penalties. The previous
comparison therefore gave TattleTots an input advantage and is superseded.

The stream now contains only fused OPIR and camera returns. For each sampled
cell, the strongest confidence from detections landing on that cell is
published; covered cells with no detection publish zero, and OPIR false
positives are preserved. `OBSERVED` now means sensor coverage, not successful
detection: OPIR covers the grid on cadence steps, while cameras cover
in-range line-of-sight cells. Coverage is independent of fire state, and the
state-independence tests include covered-but-undetected cells.

The comparison's adapter and A1 use the same canonical camera placement
formula and `max_range=10.0`. A4 now receives the comparison's camera count
and OPIR cadence explicitly, avoiding a configuration mismatch. A0 remains a
patrol-plus-OPIR architecture, while A2 intentionally uses its existing
global-state conventional-optimizer assumption; those are architecture
exposure differences, not independently placed adapter sensors.

Camera coverage is now single-sourced through `CameraTower.covers_cell`, which
checks range and line of sight. Both camera detection and adapter coverage
status use that predicate. The day/night convention is likewise shared by A1
and the adapter. When OPIR is ablated, the adapter skips the OPIR scan instead
of consuming its random draws before camera observations.

## Instrument validation

Fresh default adapter, 200 steps. The instrument-validation values below are
unchanged between the two engine pins:

| Metric | Before | After |
|---|---:|---:|
| Valid | No | Yes |
| Inferability precision | 0.000 | 0.844 |
| Decoder precision | 1.000 | 0.778 |
| Static-prior precision | 0.390 | 0.356 |
| Uniform chance | 0.100 | 0.0025 |
| Candidate locations | 10 | 400 |

The current declared-geometry decoder measures **0.778** precision (the
incumbent thermal-only decoder also measured **0.778**); the modality tier keeps
event-detection streams ahead of weather and fuel context while still honoring
their declared coordinates when they are the only evidence. The lower measured
static-prior value is a fresh
200-step remeasurement after the sensor calls consume the adapter RNG stream;
it remains the localization null. The instrument is valid and inferability is
well above the 0.25% uniform chance over the declared 20x20 frame.

When no input stream with declared geometry contains usable evidence,
`infer_report_location` abstains with `None`; TattleTots preserves its existing
spatial fallback rather than converting abstention into the origin `(0, 0)`.

## Comparison re-baseline

Default comparison configuration: 100 steps, 20x20 grid, seed 42, three
cameras, OPIR cadence 5.

| Architecture | Metric | Before ground-truth feed | Prior sensor-driven measurement (`d4cfee1`) | Current sensor-driven measurement (`cee59f9`) |
|---|---|---:|---:|---:|
| A0 Human | detections | 581 | 581 | 581 |
| A0 Human | cost | 2065.0 | 2065.0 | 2065.0 |
| A0 Human | burned cells | 399 | 399 | 399 |
| A1 Camera ML | detections | 859 | 859 | 859 |
| A1 Camera ML | cost | 1113.0 | 1113.0 | 1113.0 |
| A1 Camera ML | burned cells | 361 | 361 | 361 |
| A2 Centralized | detections | 30 | 30 | 30 |
| A2 Centralized | cost | 534.0 | 534.0 | 534.0 |
| A2 Centralized | burned cells | 17 | 17 | 17 |
| A3 Federated | detections | 2818 | 2818 | 2818 |
| A3 Federated | cost | 1292.5 | 1292.5 | 1292.5 |
| A3 Federated | burned cells | 397 | 397 | 397 |
| A4 BMA | detections | 1334 | 1335 | 1369 |
| A4 BMA | suppressions | 1 | 0 | 0 |
| A4 BMA | cost | 401.5 | 400.0 | 400.0 |
| A4 BMA | burned cells | 400 | 399 | 399 |
| A4 BMA | mean detection latency | 2.06 | 1.74 | 1.72 |
| A4 BMA | Tot detections | 184 | 170 | 208 |
| A4 BMA | OPIR detections | 1150 | 1165 | 1161 |

The A0-A3 values are unchanged because this data-path change affects the A4
adapter stream. Earlier Fire comparison results, including the old A4 values,
are superseded by this note and the JSON artifact.

### OPIR ablation RNG correction and engine remeasurement

The default 100-step OPIR-ablation arm changed after removing the disabled
OPIR scan from the adapter stream:

| Metric | Before fix | Prior after fix (`d4cfee1`) | Current after fix (`cee59f9`) |
|---|---:|---:|---:|
| Detections | 164 | 185 | 234 |
| Suppressions | 1 | 1 | 1 |
| Cost | 401.5 | 401.5 | 401.5 |
| Burned cells | 399 | 400 | 400 |
| Mean detection latency | 5.75 | 5.49 | 5.52 |
| Tot detections | 164 | 185 | 234 |
| OPIR detections | 0 | 0 | 0 |

The before-fix values were measured from source snapshot `1a2e0da` with the
current engine environment. The prior after-fix values were measured from
Fire source `42eb583` against TattleTots `d4cfee1`; the current after-fix
values were measured from the same Fire source against TattleTots
`cee59f93f6973fa7fefb2f87dbb40a8ce0095113`. The ablation has a clean
sensor-RNG path: disabling OPIR removes its stream scan rather than perturbing
subsequent camera and downstream draws.

### Thermal input density and comparison caveat

The density measurements below are scoped to TattleTots engine `d4cfee1`.
They were produced by an ad-hoc script that was never committed and no
longer exists; they are not reproducible from this repository, and this
rebaseline does not attribute them to the committed comparison harness.
Over the same 200-step external grid sequence with adapter, physics, and sensor
seed 42, the mean fraction of thermal features that were nonzero fell from
**8.0%** under the former ground-truth feed to **4.9%** under the sensor-driven
feed. The mean fraction declared `OBSERVED` was **97.33%** in both paths. The
old ground-truth arm recorded **184** Tot detections on engine `d4cfee1`,
whereas the current sensor-driven Tot path records **208** on engine
`cee59f93f6973fa7fefb2f87dbb40a8ce0095113`. The old ground-truth arm cannot be
remeasured because that code path no longer exists. No cross-engine inference
about thermal-density sensitivity is drawn here.

The default comparison burns essentially the whole 20x20 grid: the recorded
burned-area values are 399–400 of 400 cells across the A4 before/after arms
and 361–399 for A0–A3. Burned area therefore cannot discriminate architectures
under this pre-existing comparison configuration; this caveat is not caused by
the sensor-routing change.
