# AGENTS.md — AI Agent Guidelines for Scrapiron and the Bear

## Repository Purpose
FireEcology domain simulation — a testbed for TattleTots. Grid-based wildfire
model with drone Tots, competing architectures (A0-A4), and phased deployment scenarios.

## Setup
```bash
pip install -e ".[dev]"
pre-commit install
```

## Validation Commands
Run these before committing:
```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest
```

## Architecture Rules
- **Domain-specific code only** — never modify TattleTots engine or its models
- **Implement `DomainAdapter` ABC** — the adapter bridges fire sim → TattleTots
- **All architectures get the same sensors** — no strawmen
- **OPIR is always available** — it is a backstop for ALL architectures, not BMA-exclusive
- **Drones are physical Tots** — they have body plans (hardware) and genomes (behavior)
- **Body plans don't mutate** — hardware is fixed; behavioral traits evolve
- **Never modify tests to make them pass** — fix the implementation

## Key Files
| File | Purpose |
|------|---------|
| `src/fire_ecology/adapter/fire_adapter.py` | TattleTots DomainAdapter implementation |
| `src/fire_ecology/environment/fire.py` | Fire spread cellular automaton |
| `src/fire_ecology/drones/drone_genome.py` | Heritable drone behavioral traits |
| `src/fire_ecology/architectures/a2_centralized.py` | Strongest conventional competitor |
| `src/fire_ecology/scenarios/phased_deployment.py` | 4-phase hardware rollout |
| `src/fire_ecology/metrics/fire_metrics.py` | Spec §9 falsification metrics |

## Module Dependency Order
```
environment → sensors → drones → users → architectures → adapter → metrics → scenarios → cli
```

## Spec Documents
- `fire_tots_spec_v2.md` — Domain specification with all requirements
- `domain_master_plan_v2.md` — Cross-domain architecture comparison plan

## PR Requirements
- All ruff checks pass
- mypy strict passes on src/
- All tests pass (including smoke tests)
- New features include tests
- Update README if adding new scenarios or architectures
