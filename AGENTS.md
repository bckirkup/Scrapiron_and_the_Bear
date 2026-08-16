# AGENTS.md — AI Agent Guidelines for Scrapiron and the Bear

## Repository Purpose
FireEcology domain simulation — a testbed for TattleTots. Grid-based wildfire
model with drone Tots, competing architectures (A0-A4), and phased deployment scenarios.

## Setup
```bash
uv sync --locked --no-build --no-binary-package fire-ecology --no-binary-package domain-runner --no-binary-package tattletots --extra dev
uv run --no-sync --no-build pre-commit install
```

## Before Editing
- Read `.agents/skills/sonar-quality/SKILL.md` before writing or changing code.

## Validation Commands
Run these before committing:
```bash
uv run --no-sync --no-build pre-commit run --all-files
uv run --no-sync --no-build python scripts/sonar_guard.py src tests scripts baselines
uv run --no-sync --no-build python scripts/sonar_guard.py --workflows .github/workflows
uv run --no-sync --no-build ruff check src/ tests/ scripts/ baselines/
uv run --no-sync --no-build ruff format --check src/ tests/ scripts/ baselines/
uv run --no-sync --no-build mypy src/
uv run --no-sync --no-build pytest --strict-markers -ra
uv run --no-sync --no-build pytest -m smoke --tb=short -q
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
| `src/fire_ecology/adapter/fire_adapter.py` | TattleTots DomainAdapter + COP dispatch hooks (`score_relevance` → band alignment) |
| `src/fire_ecology/runner.py` | domain-runner hooks (`FireDomainHooks`, `run_fire_simulation`) |
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
