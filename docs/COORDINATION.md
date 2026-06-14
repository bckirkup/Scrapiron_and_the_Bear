# Cross-Repository Coordination Guide

This document explains how Scrapiron_and_the_Bear (FireEcology) integrates with TattleTots and the sibling domain repos.

## Repository Ecosystem

| Repository | Role | Package |
|------------|------|---------|
| **TattleTots** | Agent ecology engine (domain-agnostic) | `tattletots` |
| **Coral_Key_in_Three_Hour_Epochs** | ReefWatch fishery domain adapter | `coral-key` |
| **Xylella_SPQR** | GrainGuard agriculture domain adapter | `grain-guard` |
| **Scrapiron_and_the_Bear** (this repo) | FireEcology wildfire domain adapter | `fire-ecology` |

## How FireEcology Connects to TattleTots

FireEcology implements the `DomainAdapter` ABC from TattleTots:

```python
from tattletots.interface.domain_adapter import DomainAdapter

class FireEcologyAdapter(DomainAdapter):
    def get_streams(self) -> list[Stream]: ...      # OPIR, cameras, weather, fuel moisture
    def get_users(self) -> list[User]: ...          # Lookout, Dispatcher, Commander
    def step(self, time_step: int) -> None: ...     # Advance fire + weather + sensors
    def get_ground_truth(self, time_step: int) -> bool: ...  # Fire active?
    def compute_costs(self, ...) -> dict[str, float]: ...    # Suppression + damage costs
```

## Installation for Coordinated Use

```bash
# Install TattleTots first (engine dependency)
pip install -e /path/to/TattleTots[dev]

# Install this repo
pip install -e ".[dev]"

# Optionally install sibling domains for cross-comparison
pip install -e /path/to/Coral_Key_in_Three_Hour_Epochs[dev]
pip install -e /path/to/Xylella_SPQR[dev]
```

## Running Modes

### Standalone (domain-only, no agent ecology)

```bash
fire-ecology --steps 200 --verbose --json
```

Exercises the wildfire simulation (fire spread, weather, sensors) without TattleTots agents.

### Integrated (domain + TattleTots agent ecology)

```bash
python scripts/run_with_tattletots.py \
    --config configs/tattletots_integration.json \
    --output integrated_results.json \
    --verbose
```

Runs the full loop: domain generates sensor streams → Tot agents compress/escalate → trust/evolution dynamics → cost accounting.

## Configuration

The integrated config (`configs/tattletots_integration.json`) has two sections:

- **`simulation`**: TattleTots engine params (population size, mutation rate, trust dynamics)
- **`domain`**: FireEcology params (grid size, thermal stream dimension, seed)

### Key Parameters to Tune

| Parameter | Section | Effect |
|-----------|---------|--------|
| `initial_population` | simulation | Number of starting Tot agents |
| `max_stream_dim` | simulation | Per-agent input cap (keep ≤30 for performance) |
| `grid_rows`/`grid_cols` | domain | Terrain resolution |
| `max_thermal_dim` | domain | Thermal stream dimension (null = default) |
| `steps` | domain | Simulation length |
| `seed` | domain | Reproducibility |

## Output Format

Integrated runs produce unified JSON (see TattleTots `docs/COORDINATION.md` for full schema).

Domain-specific metrics in `domain_metrics`:

```json
{
  "mean_detection_latency": 2.1,
  "suppression_success_rate": 0.72,
  "false_dispatch_rate": 0.05,
  "opir_rescue_rate": 0.15,
  "total_cost": 4800.0,
  "total_fires_detected": 35.0,
  "total_escalations": 28.0,
  "total_burned_area": 120.0
}
```

## Cross-Domain Comparison

All domain repos produce the same top-level structure. Compare across domains:

```python
from tattletots.output_schema import SimulationOutput

coral = SimulationOutput.read_json("coral_results.json")
fire = SimulationOutput.read_json("fire_results.json")
grain = SimulationOutput.read_json("grain_results.json")

# Same metrics available for each
for r in [coral, fire, grain]:
    print(f"{r.run_summary.domain}: cost={r.cost_metrics.total_cost:.0f} "
          f"precision={r.ecology_metrics.precision:.2%}")
```

## Relationship to Sibling Repos

Each domain repo is structurally parallel:
- `src/<package>/adapter/` — DomainAdapter implementation
- `scripts/run_with_tattletots.py` — Integrated runner
- `configs/tattletots_integration.json` — Default integrated config
- `docs/COORDINATION.md` — This file

The domains share no code with each other — only with TattleTots via the `DomainAdapter` interface. This ensures each domain can evolve independently while maintaining compatible outputs.
