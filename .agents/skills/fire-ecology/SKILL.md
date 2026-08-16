---
name: fire-ecology-development
description: Guide for developing and testing the FireEcology domain simulation for TattleTots.
---

# FireEcology Development Skill

## Setup
```bash
uv sync --locked --no-build --no-binary-package fire-ecology --no-binary-package domain-runner --no-binary-package tattletots --extra dev
uv run --no-sync --no-build pre-commit install
```

## Running the Simulation

```bash
# Domain physics only (no TattleTots required)
uv run --no-sync --no-build fire-ecology sim --layer domain_only --steps 50 --grid-rows 10 --grid-cols 10 --verbose

# Batch sweeps
uv run --no-sync --no-build fire-ecology batch --config configs/batch_example.json

# Full agent ecology + COP dispatch
uv run --no-sync --no-build fire-ecology sim --layer tattletots --config configs/tattletots_integration.json

# Legacy quick run
uv run --no-sync --no-build fire-ecology --steps 200 --verbose
```

## Testing
```bash
# All tests
uv run --no-sync --no-build pytest

# Smoke tests only
uv run --no-sync --no-build pytest -m smoke

# Specific module
uv run --no-sync --no-build pytest tests/test_fire.py -v

# With coverage
uv run --no-sync --no-build pytest --cov=fire_ecology --cov-report=term-missing
```

## Linting & Type Checking
```bash
uv run --no-sync --no-build ruff check src/ tests/
uv run --no-sync --no-build ruff format --check src/ tests/
uv run --no-sync --no-build mypy src/
```

## Key Architecture Notes

### Module Dependency Order
```
environment → sensors → drones → users → architectures → adapter → metrics → scenarios → cli
```

### Adding a New Sensor
1. Create `src/fire_ecology/sensors/my_sensor.py` with a Pydantic model
2. Add observation method returning `np.ndarray`
3. Wire into `adapter/fire_adapter.py`:
   - Add to `__init__` placement
   - Add to `_setup_streams()` dimensionality
   - Add to `_update_streams()` data flow
4. Add tests in `tests/test_sensors.py`
5. Update `sensors/__init__.py`

### Adding a New Architecture
1. Subclass `Architecture` from `architectures/base.py`
2. Implement `step()` returning `ArchitectureResult`
3. Implement `reset()`
4. Give it the same sensor/hardware access
5. Add tests in `tests/test_architectures.py`
6. Update `architectures/__init__.py`

### TattleTots Integration
The adapter (`adapter/fire_adapter.py`) implements `DomainAdapter`:
- `get_streams()` → returns thermal, weather, fuel moisture streams
- `get_users()` → returns 3 fire-domain user profiles
- `step(time_step)` → advances fire sim, updates sensor streams
- `get_ground_truth(time_step)` → True if any cell is burning
- `get_active_locations(time_step)` → returns `(row, col)` of all burning cells
- `infer_report_location(stream_data, stream_labels)` → finds peak in thermal stream → maps to grid `(row, col)`
- `score_relevance(signal, user)` → band-aligned role relevance via `tattletots.engine.relevance`
- `compute_costs(...)` → surveillance + response + damage costs
- `get_responder_user_id()` → user authorized for COP dispatch
- `dispatch_and_judge_responses(targets, time_step)` → execute suppression, return outcomes

**Note:** The integration loop uses `world.set_event_state(adapter.get_active_locations(step))` (not `set_ground_truth`). The engine verifies report correctness per-location. Agents must not read `User.trust`.

### Baselines

Standalone baseline comparison files live in `baselines/`:
- `run_fire_ecology_baselines.py` — Parameter scan runner for A0-A3 architectures
- `fire_ecology_baselines_config.json` — Scan configuration
- `fire_ecology_baselines_results.zip` — Pre-computed results

## Integrated Mode (Full Agent Ecology)

```bash
uv run --no-sync --no-build fire-ecology sim --layer tattletots --config configs/tattletots_integration.json --output results.json --verbose

# Legacy wrapper
uv run --no-sync --no-build python scripts/run_with_tattletots.py \
    --config configs/tattletots_integration.json \
    --output results.json --verbose
```

Output conforms to `tattletots.output_schema.SimulationOutput` (unified JSON).
See `docs/COORDINATION.md` for coordination with sibling repos.

## GPU Acceleration

```bash
uv sync --locked --no-build --no-binary-package fire-ecology --no-binary-package domain-runner --no-binary-package tattletots --extra dev --extra gpu
```

Set `"use_gpu": true` in the `"simulation"` section of the integration config.
Falls back silently to NumPy if CuPy or CUDA is unavailable.

## Parameter Scans

Generate config variants and run in parallel for large sweeps:

```bash
uv run --no-sync --no-build python scripts/run_with_tattletots.py --config <variant>.json --output results/<name>.json
```

Key domain parameters to sweep: `grid_rows`, `grid_cols`, `ignition_probability`,
`wind_speed`, `steps`, `seed`.

Load results:
```python
from tattletots.output_schema import SimulationOutput
result = SimulationOutput.model_validate_json(path.read_text())
```

## Measurement Harness Testing (designed reporter / margin arms)

Everything runs through uv; never use plain `uv run` or `pip install`:
```bash
uv run --no-sync --no-build python scripts/run_designed_reporter_experiment.py \
  --steps 40 --seeds 42 43 44 --grid-rows 12 --grid-cols 12 --n-cameras 2 \
  --base-ignition-rate 0.01 --initial-population 6 --max-population 12 \
  --jobs 2 --docs-dir /tmp/dr_check
```
A short config like the above finishes in ~20 s and writes the same artifact set as
the committed 200-step / 20-seed run: `<docs-dir>/designed_reporter_measurement.json`,
`.md`, and one `SimulationOutput` per arm under `<docs-dir>/designed_reporter/<arm>.json`.
Always point `--docs-dir` at a scratch dir so committed artifacts are not clobbered.

Testing gotchas worth remembering:
- **Per-arm JSONs are not byte-reproducible**: `SimulationOutput` stamps a `timestamp`
  field, so determinism checks must compare the top-level JSON (`nulls`, `arms`,
  `margin`, `per_seed`) or diff arm files while ignoring `.timestamp`.
- `--jobs N` must not change any number; comparing a `--jobs 1` run to a `--jobs 2`
  run is a cheap nondeterminism probe.
- **Tampering with a reporter policy's defaults**: the policy is a dataclass, so
  patching `FireThermalEvidenceReporterPolicy.confidence_floor` on the class does
  nothing to new instances. Re-register the factory instead, and restore it after:
  `register_reporter_policy(FIRE_REPORTER_POLICY_NAME, lambda: FireThermalEvidenceReporterPolicy(confidence_floor=0.0))`.
- **Proving the confidence floor is load-bearing needs a no-fire window**: with fires
  present a camera detection (~0.72) dominates `max()`, so admitting the 0.3 OPIR
  false positives changes nothing. Run `--base-ignition-rate 0.0`: at floor 0.45 the
  designed arms emit zero reports (unscorable); at floor 0.0 they emit false reports.
- A zero-ignition window is a good crash/edge probe: `nulls.event_steps == 0`, all
  nulls collapse to 0.0, and evolved arms may still report (0 % precision) while the
  designed arms land in `margin.unscorable_arms`.
- Ground-truth leakage can be checked at runtime by spying on `decide(context)` and
  listing `dir(context)`; the engine only exposes `streams`, `observation`,
  `signal_vector`, `anomaly_score`, `escalation_threshold`, `location_frame`, `time_step`.
- A `RuntimeWarning: overflow encountered in square` from `numpy/_core/_methods.py`
  shows up in these runs and in unrelated suites (e.g. `tests/test_comparison.py`);
  it is pre-existing and non-fatal, not a signal from the harness under test.
