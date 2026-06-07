# Contributing to Scrapiron and the Bear

## Rule 0: Patch-Not-Report

If you find a bug, **fix it and submit a PR** rather than filing an issue.
Include a test that fails before your fix and passes after.

## Development Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Code Conventions

- Python 3.11+ with strict typing (mypy strict mode)
- Pydantic v2 for all data models
- NumPy for numerical operations
- All modules use `from __future__ import annotations`
- Ruff for linting and formatting (line length 100)
- Tests use pytest with `smoke` and `slow` markers

## Before Submitting

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest
```

All checks must pass.

## Architecture Principles

1. **No strawmen**: Every competing architecture gets the same sensors and hardware.
2. **OPIR is universal**: It is a backstop available to all architectures.
3. **Drones are Tots**: They have fixed body plans and evolvable behavioral genomes.
4. **Domain adapter pattern**: All TattleTots integration goes through `DomainAdapter`.
5. **Phased deployment**: Evaluate under incremental hardware rollout, not steady state.

## Adding New Architectures

1. Subclass `Architecture` from `architectures/base.py`
2. Implement `step()` and `reset()`
3. Give the architecture the same sensor access as existing ones
4. Add tests in `tests/test_architectures.py`
5. Update `architectures/__init__.py`

## Adding New Sensors

1. Create a new model in `sensors/`
2. Add stream output to the adapter in `adapter/fire_adapter.py`
3. Add tests
4. Update `sensors/__init__.py`
