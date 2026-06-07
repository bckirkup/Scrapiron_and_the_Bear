# Scrapiron and the Bear — FireEcology Domain Simulation

A testbed for [TattleTots](https://github.com/bckirkup/TattleTots): autonomous fire-regime
management with physical drone Tots.

## Overview

FireEcology is a grid-based wildfire simulation that tests whether a self-adaptive
drone/sensor ecology (the TattleTots "Biomorphic Metabolic Architecture") can reduce
total expected fire-regime cost compared to competent centralized alternatives.

The simulation covers:
- **Active fire detection** and hotspot discovery
- **Dry-fuel monitoring** and fire weather indexing
- **Controlled-burn monitoring** and escape detection
- **Autonomous drone suppression** with physical Tot body plans
- **Human-response escalation** with attention-budget users
- **OPIR satellite backstop** available to all architectures

## Architecture

```
src/fire_ecology/
├── environment/     # Terrain, fuel, weather, fire dynamics
│   ├── terrain.py   # Elevation, slope, aspect, barriers
│   ├── fuel.py      # Fuel type, load, moisture
│   ├── weather.py   # Temperature, humidity, wind, precipitation
│   └── fire.py      # Ignition, spread, suppression (cellular automaton)
├── sensors/         # Sensor models
│   ├── opir.py      # OPIR/satellite thermal (always-available backstop)
│   ├── camera_tower.py  # Fixed camera towers with LOS
│   ├── weather_station.py  # Localized weather observations
│   └── fuel_moisture.py    # Ground fuel moisture probes
├── drones/          # Physical Tot body plans
│   ├── body_plan.py     # Hardware templates (scout, strike, relay, hybrid)
│   ├── drone_genome.py  # Heritable behavioral traits
│   └── drone_state.py   # Runtime state (battery, water, position)
├── users/           # Fire-domain user profiles
│   └── fire_users.py   # Sector Mgr, Fire Ops Chief, Burn Mgr
├── architectures/   # Competing management approaches (A0-A3)
│   ├── base.py      # Architecture ABC
│   ├── a0_human.py  # Human/manual baseline
│   ├── a1_camera_ml.py  # Camera network + ML
│   ├── a2_centralized.py  # Centralized drone fleet optimizer
│   └── a3_federated.py   # Federated edge network
├── adapter/         # TattleTots DomainAdapter implementation
│   └── fire_adapter.py  # Bridges fire sim → TattleTots engine
├── metrics/         # Fire-domain metrics (spec §9)
│   └── fire_metrics.py  # Detection latency, OPIR rescue rate, costs
├── scenarios/       # Deployment configurations
│   └── phased_deployment.py  # 4-phase hardware rollout (spec §8)
└── cli.py           # Command-line entrypoint
```

## Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Development Commands

```bash
# Run all tests
pytest

# Run smoke tests only
pytest -m smoke

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# Run simulation
fire-ecology --steps 200 --verbose
fire-ecology --steps 100 --json
```

## Competing Architectures

All architectures receive the same sensors, data streams, and hardware:

| ID | Architecture | Description |
|----|-------------|-------------|
| A0 | Human Baseline | Lookouts, patrols, conventional dispatch |
| A1 | Camera/ML Network | ALERT-style cameras + smoke ML + human verification |
| A2 | Centralized Optimizer | Global state estimation + MPC drone routing |
| A3 | Federated Edge | Local nodes with limited cross-region fusion |
| A4 | BMA / TattleTots | Self-organizing drone/sensor ecology (via TattleTots engine) |

## Phased Deployment (Spec §8)

| Phase | Months | Drones | Cameras | Weather Stn | Coverage |
|-------|--------|--------|---------|-------------|----------|
| 0 | 0-6 | 5 | 3 | 4 | 50 km² |
| 1 | 6-12 | 15 | 8 | 4 | 100 km² |
| 2 | 12-18 | 30 | 12 | 6 | 200 km² |
| 3 | 18-24 | 30 | 12 | 8 | 300 km² |

## Metrics (Spec §9)

- Detection latency (active fires)
- Hotspot lead time
- Dry-spot prediction precision/recall
- Controlled burn escape detection latency
- Autonomous suppression success rate
- False dispatch rate
- OPIR rescue rate
- Drone sortie / logistics cost
- Total cost: surveillance + response + damage + attention

## License

Apache-2.0 — see [LICENSE](LICENSE).
