# Fire sensor-driven stream re-baseline

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

## Instrument validation

Fresh default adapter, 200 steps:

| Metric | Before | After |
|---|---:|---:|
| Valid | No | Yes |
| Inferability precision | 0.000 | 0.844 |
| Decoder precision | 1.000 | 0.778 |
| Static-prior precision | 0.390 | 0.356 |
| Uniform chance | 0.100 | 0.0025 |
| Candidate locations | 10 | 400 |

The decoder drop is expected: sparse, noisy sensor returns replace the former
direct intensity feed. The lower measured static-prior value is a fresh
200-step remeasurement after the sensor calls consume the adapter RNG stream;
it remains the localization null. The instrument is valid and inferability is
well above the 0.25% uniform chance over the declared 20x20 frame.

When the sensor-driven vector is all zero, `infer_report_location` retains its
existing `argmax` behavior and returns the first declared thermal sample. This
is a real abstention/degeneracy characteristic of the unchanged decoder, not a
metadata workaround.

## Comparison re-baseline

Default comparison configuration: 100 steps, 20x20 grid, seed 42, three
cameras, OPIR cadence 5.

| Architecture | Metric | Before | After | Change |
|---|---|---:|---:|---:|
| A0 Human | detections | 581 | 581 | 0 |
| A0 Human | cost | 2065.0 | 2065.0 | 0.0 |
| A0 Human | burned cells | 399 | 399 | 0 |
| A1 Camera ML | detections | 859 | 859 | 0 |
| A1 Camera ML | cost | 1113.0 | 1113.0 | 0.0 |
| A1 Camera ML | burned cells | 361 | 361 | 0 |
| A2 Centralized | detections | 30 | 30 | 0 |
| A2 Centralized | cost | 534.0 | 534.0 | 0.0 |
| A2 Centralized | burned cells | 17 | 17 | 0 |
| A3 Federated | detections | 2818 | 2818 | 0 |
| A3 Federated | cost | 1292.5 | 1292.5 | 0 |
| A3 Federated | burned cells | 397 | 397 | 0 |
| A4 BMA | detections | 1334 | 1335 | +1 |
| A4 BMA | suppressions | 1 | 0 | -1 |
| A4 BMA | cost | 401.5 | 400.0 | -1.5 |
| A4 BMA | burned cells | 400 | 399 | -1 |
| A4 BMA | mean detection latency | 2.06 | 1.74 | -0.32 |
| A4 BMA | Tot detections | 184 | 170 | -14 |
| A4 BMA | OPIR detections | 1150 | 1165 | +15 |

The A0-A3 values are unchanged because this data-path change affects the A4
adapter stream. Earlier Fire comparison results, including the old A4 values,
are superseded by this note and the JSON artifact.
