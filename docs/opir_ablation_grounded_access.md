# OPIR-ablated measurement: does grounded raw-stream access raise agent-only detection?

**Answer: no.** With the OPIR backstop ablated, agent-only detection *falls*
monotonically in the arm means as `grounded_input_fraction` rises
(0.0248 → 0.0180 → 0.0148 cell-step detection rate; cell recall 0.329 → 0.246 → 0.206),
while the mechanism the TattleTots fix targets does engage: grounded yield share
rises 0.10 → 0.82 → 1.00 and the engine stops flagging
`grounded_yield_share_below_minimum` initiation degeneracy. Report precision
edges up (0.142 → 0.156 → 0.180 mean) but stays under the instrument-level
static-prior null of 0.356, i.e. below what a fixed guess at the most common
event location would score.

## Why the ablation was needed

`architectures/a4_bma.py` used to append the OPIR backstop to `detections`
unconditionally, so every domain detection metric survived total Tot extinction.
The reported detection rate in the assisted arms is ~0.195–0.203 in every arm and
is almost entirely OPIR: at `grounded_input_fraction=0.0`, seed 42, agents
contributed 129 detection cell-steps while OPIR contributed 1183. That path
cannot falsify anything about the agents.

`EcologyMeasurementOptions.ablate_opir_backstop` (default `False`, so existing
behavior is unchanged) withholds the backstop from the reported detections and
books the withheld hits as `opir_shadow_detections`, keeping both numbers
visible. The OPIR satellite still feeds the thermal stream and still consumes its
random draws, so an ablated arm starts from the same draws as its baseline.

The ablation is verified in `tests/test_opir_ablation.py`: with every
Tot forced to `LifecycleStage.DEAD`, the ablated path reports **0** detections
and 0 reports, while the assisted path still reports detections from OPIR alone.

## Which numbers are agent-only and which are OPIR-assisted

| number | meaning |
|---|---|
| `agent_only_detection_rate`, `agent_only_step_detection_rate`, `agent_only_cell_recall` | **agent-only.** Derived from Tot reports only; zero when the ecology is extinct. |
| `report_precision`, `report_false_alarm_rate` | **agent-only.** Engine report bookkeeping (`correct_reports` / `false_alarms`), never touched by OPIR. |
| `reported_detection_rate` | **path-dependent.** Equals the agent-only rate in `*_opir_ablated` arms; includes the backstop in `*_opir_assisted` arms. |
| `opir_shadow_detection_cell_steps`, `opir_assisted_detection_rate_counterfactual` | **OPIR-assisted.** What the backstop would have added / did add. |
| `attention_solvent_fraction`, `mean_per_capita_attention_capacity`, `grounded_yield_share` | **agent-only** (TattleTots ecology telemetry; no OPIR term). |

Suppression couples the two paths: an assisted arm suppresses the OPIR-detected
cells, so after the first step where the detection sets differ the fire state
diverges. Ablated and assisted arms at the same seed are therefore *not* two
views of one trajectory; they are two runs from the same seed and initial state.
Compare arms within a path, not across paths. Agent-only columns in the assisted
table are shown only to confirm the ablation does not distort agent behavior.

## Setup

- Scenario: 20×20 grid, 200 steps, 10 drones, 3 cameras, OPIR cadence 5, hybrid body plan.
- Seeds 42–46, identical across arms; every arm uses the same scenario config.
- Engine: TattleTots branch `devin/1786813382-grounded-stream-access`
  (PR [#55](https://github.com/bckirkup/TattleTots/pull/55)) installed into the venv only —
  the committed lockfile still pins the released tattletots rev.
- Reproduce: `uv run --no-sync --no-build python scripts/measure_grounded_access.py
  --steps 200 --seeds 42 43 44 45 46 --instrument-steps 200 --output-dir docs/grounded_access`

`SimulationOutput` JSON: `docs/grounded_access/<arm>_seed<seed>.json`,
flat summary with per-arm means: `docs/grounded_access/summary.json`.

## Nulls

Report precision is meaningless without them. Both engine-level nulls travel with
every arm (`null_chance_precision`, `null_static_prior_precision` below), and the
instrument-level nulls come from
`tattletots.interface.instrument.validate_instrument` on `FireEcologyAdapter`
(200 steps, per seed; seed 42 shown):

| instrument null | value |
|---|---|
| `null_chance_baseline` (uniform over 400 candidate cells) | 0.0025 |
| `null_static_prior_baseline` (most common event location) | 0.3556 |
| `inferability_precision` | 0.8444 |
| `decoder_precision` | 0.7778 |
| event steps / measured steps | 45 / 200 |

All instrument checks pass, so the harness can in principle see the events; the
agents' 0.14–0.18 report precision is a competence result, not an instrument
artifact. It beats the uniform null (0.0025) and the engine's own chance null
(~0.080) but loses to the static-prior null (0.3556): a fixed guess at the
single most fire-prone cell would outscore the ecology.

## OPIR-ablated arms (agent-only detections)

Mean over seeds 42–46, `[min, max]` across seeds.

| metric | gif0.00_opir_ablated | gif0.34_opir_ablated | gif0.67_opir_ablated |
|---|---|---|---|
| `agent_only_detection_rate` | 0.0248 [0.0158, 0.0395] | 0.0180 [0.0141, 0.0265] | 0.0148 [0.0075, 0.0220] |
| `agent_only_step_detection_rate` | 0.8291 [0.7755, 0.8936] | 0.8036 [0.7234, 0.8491] | 0.7208 [0.4915, 0.9184] |
| `agent_only_cell_recall` | 0.3293 [0.2450, 0.5425] | 0.2463 [0.2050, 0.3133] | 0.2056 [0.1128, 0.3025] |
| `reported_detection_rate` | 0.0248 [0.0158, 0.0395] | 0.0180 [0.0141, 0.0265] | 0.0148 [0.0075, 0.0220] |
| `report_precision` | 0.1418 [0.0860, 0.2000] | 0.1558 [0.1103, 0.2432] | 0.1795 [0.0898, 0.2674] |
| `report_false_alarm_rate` | 0.8582 [0.8000, 0.9140] | 0.8442 [0.7568, 0.8897] | 0.8205 [0.7326, 0.9102] |
| `null_chance_precision` | 0.0800 [0.0799, 0.0800] | 0.0800 [0.0799, 0.0800] | 0.0800 [0.0800, 0.0800] |
| `null_static_prior_precision` | 0.0668 [0.0360, 0.1336] | 0.0645 [0.0453, 0.0951] | 0.0817 [0.0372, 0.1395] |
| `attention_solvent_fraction` | 0.4743 [0.4174, 0.6322] | 0.4872 [0.4533, 0.5712] | 0.4438 [0.3972, 0.5012] |
| `mean_per_capita_attention_capacity` | 2.0059 [1.7182, 2.4779] | 1.8384 [1.4198, 2.7511] | 2.0962 [1.4470, 3.4605] |
| `grounded_yield_share` | 0.1026 [0.0150, 0.2407] | 0.8207 [0.7044, 0.8953] | 1.0000 [1.0000, 1.0000] |
| `effective_grounded_yield_share` | 0.1643 [0.0294, 0.3732] | 0.8515 [0.7697, 0.9074] | 1.0000 [1.0000, 1.0000] |
| `parent_child_reproductive_correlation` | -0.0364 [-0.1686, 0.0637] | 0.0722 [-0.0842, 0.2875] | 0.0123 [-0.1892, 0.2409] |

Paired per-seed differences in `agent_only_detection_rate` versus the
`grounded_input_fraction=0.0` arm at the same seed:

| seed | 0.34 − 0.00 | 0.67 − 0.00 |
|---|---|---|
| 42 | -0.0033 | -0.0108 |
| 43 | +0.0058 | -0.0132 |
| 44 | -0.0113 | -0.0058 |
| 45 | -0.0255 | -0.0261 |
| 46 | +0.0003 | +0.0063 |

4 of 5 seeds fall at 0.67 and 3 of 5 at 0.34. The direction is down or flat; with
5 seeds the honest reading is "no improvement", not "a reliable decrease".

## OPIR-assisted arms (legacy path, for reference)

| metric | gif0.00_opir_assisted | gif0.34_opir_assisted | gif0.67_opir_assisted |
|---|---|---|---|
| `agent_only_detection_rate` | 0.0237 [0.0130, 0.0466] | 0.0189 [0.0150, 0.0263] | 0.0133 [0.0064, 0.0195] |
| `agent_only_cell_recall` | 0.3035 [0.2000, 0.5725] | 0.2611 [0.2050, 0.3375] | 0.1940 [0.1025, 0.2725] |
| `reported_detection_rate` (OPIR-assisted) | 0.2034 [0.1921, 0.2244] | 0.1979 [0.1914, 0.2041] | 0.1946 [0.1914, 0.1983] |
| `report_precision` | 0.1481 [0.0400, 0.2128] | 0.1700 [0.0891, 0.2500] | 0.1345 [0.1098, 0.1526] |
| `report_false_alarm_rate` | 0.8519 [0.7872, 0.9600] | 0.8300 [0.7500, 0.9109] | 0.8655 [0.8474, 0.8902] |
| `null_chance_precision` | 0.0800 [0.0799, 0.0800] | 0.0800 [0.0800, 0.0800] | 0.0800 [0.0800, 0.0800] |
| `null_static_prior_precision` | 0.0913 [0.0372, 0.1411] | 0.0653 [0.0538, 0.0791] | 0.0872 [0.0488, 0.1413] |
| `grounded_yield_share` | 0.1629 [0.0143, 0.3992] | 0.7906 [0.5951, 0.8970] | 1.0000 [1.0000, 1.0000] |

The assisted `reported_detection_rate` is flat at ~0.195–0.203 across arms while
agent-only detection varies by 2×. That flatness is the masking: it tracks OPIR
coverage, not the ecology.

## Solvency, grounded yield, and reproduction

- **Per-capita attention solvency** is unchanged by grounded access:
  `attention_solvent_fraction` 0.474 / 0.487 / 0.444 and
  `mean_per_capita_attention_capacity` 2.01 / 1.84 / 2.10, all with overlapping
  per-seed ranges. Attention is not the binding constraint being relieved.
- **Grounded yield share** is where the fix lands: 0.103 → 0.821 → 1.000
  (effective: 0.164 → 0.852 → 1.000), monotone in every seed. At 0.0 all five
  seeds are flagged `grounded_yield_share_below_minimum`; at 0.34 and 0.67 no
  seed is flagged for initiation degeneracy at all. So grounded access fixes the
  *input diet*, and the diet was not what limited detection.
- **Parent-child reproductive correlation** (Pearson between a parent's and its
  child's offspring counts) is computable but near zero everywhere:
  -0.036 / +0.072 / +0.012, with per-seed ranges spanning zero. Reproductive
  success is not heritable in these runs, which is consistent with the
  population sitting at its `max_population=50` cap in every arm and seed
  (`final_population = 50` throughout, no extinction).

## Caveats

- 5 seeds per arm; per-seed ranges overlap for every metric except grounded
  yield share. Treat the detection decline as "no rise", not a measured decline.
- Populations saturate the cap in every arm, so selection pressure is
  reproduction-limited rather than detection-limited; a lower cap or a costlier
  attention budget would be a better setting to test whether grounded evidence
  can pay for itself.
- `_tot_detections` maps each report onto the highest-intensity active cell, so
  agent-only detection rate measures report *volume* aimed at real fire more than
  independent localization. `report_precision` (from engine bookkeeping) is the
  cleaner competence number, and it is below the static-prior null.
- `tattletots.engine.escalation.normalize_anomaly` emits `RuntimeWarning:
  overflow encountered in square` during these runs (engine-side; not modified
  here).
