---
name: fire-ecology-development
description: Guide for developing and testing the FireEcology domain simulation for TattleTots.
---

# FireEcology Development Skill

## Setup
```bash
cd /home/ubuntu/repos/Scrapiron_and_the_Bear
pip install -e ".[dev]"
pre-commit install
```

## Running the Simulation
```bash
# Quick test run
fire-ecology --steps 50 --grid-rows 10 --grid-cols 10 --verbose

# Full simulation with JSON output
fire-ecology --steps 400 --json

# Custom seed for reproducibility
fire-ecology --steps 200 --seed 123 --verbose
```

## Testing
```bash
# All tests
pytest

# Smoke tests only
pytest -m smoke

# Specific module
pytest tests/test_fire.py -v

# With coverage
pytest --cov=fire_ecology --cov-report=term-missing
```

## Linting & Type Checking
```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
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
- `score_relevance(signal, user)` → dot-product relevance
- `compute_costs(...)` → surveillance + response + damage costs
