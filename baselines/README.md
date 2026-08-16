# Baseline Comparisons (Without TattleTots)

Parameter scans using **only** conventional baseline architectures (A0–A3), no TattleTots agent ecology.

## Run from workspace root

All repos must be siblings under a common workspace root (e.g. `D:\TotsFiles\`):

```bash
cd D:\TotsFiles

# Smoke test
uv run --no-sync --no-build --project Scrapiron_and_the_Bear python Scrapiron_and_the_Bear/baselines/run_fire_ecology_baselines.py --smoke-test

# Full scan (2,160 runs) with multiprocessing
uv run --no-sync --no-build --project Scrapiron_and_the_Bear python Scrapiron_and_the_Bear/baselines/run_fire_ecology_baselines.py --workers 8
```

Parallel mode uses **ProcessPoolExecutor** (separate Python worker processes). You should see multiple `python.exe` jobs in Task Manager.

## Files

| File | Purpose |
|------|---------|
| `run_fire_ecology_baselines.py` | Parameter scan runner |
| `fire_ecology_baselines_config.json` | Factor levels, seeds, steps |
| `fire_ecology_baselines_results.zip` | Pre-computed results (optional) |

## Shared utilities

Multiprocessing helpers live in `TattleTots/Large Experiments/baseline_parallel.py` and are auto-discovered at runtime.

## Prerequisites

```bash
uv sync --locked --no-build --no-binary-package fire-ecology --no-binary-package domain-runner --no-binary-package tattletots --extra dev
```

After editing domain code, the editable install picks up changes in workers.
