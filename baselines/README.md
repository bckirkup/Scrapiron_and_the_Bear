# Baseline Comparisons (Without TattleTots)

Parameter scans using **only** conventional baseline architectures (A0–A3), no TattleTots agent ecology.

## Run from workspace root

All repos must be siblings under a common workspace root (e.g. `D:\TotsFiles\`):

```bash
cd D:\TotsFiles

# Smoke test
python Scrapiron_and_the_Bear/baselines/run_fire_ecology_baselines.py --smoke-test

# Full scan (2,160 runs) with multiprocessing
python Scrapiron_and_the_Bear/baselines/run_fire_ecology_baselines.py --workers 8
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
pip install -e TattleTots[dev]
pip install -e Scrapiron_and_the_Bear[dev]
```

After editing domain code, reinstall the fire ecology package so workers pick up changes.
