# Baseline Comparisons (Without TattleTots)

This directory contains scripts and results for running the Fire Ecology simulation
using **only** the conventional baseline management architectures (A0-A3), without
the TattleTots agent ecology.

## Contents

| File | Purpose |
|------|---------|
| `run_fire_ecology_baselines.py` | Parameter scan runner sweeping deployment phase, sensor dropout, drone fleet size, ignition rate, and weather volatility |
| `fire_ecology_baselines_config.json` | Scan configuration (factors, seeds, steps) |
| `fire_ecology_baselines_results.zip` | Pre-computed results from a full parameter scan |

## Usage

These scripts are designed to run from a workspace root that has all domain repos
installed. They depend on `baseline_parallel` (a shared utility in the TattleTots
`Large Experiments/` directory).

```bash
# From the workspace root (parent of all repos):
python Scrapiron_and_the_Bear/baselines/run_fire_ecology_baselines.py --smoke-test
```

## Relationship to TattleTots

These baselines serve as the **control group** for evaluating TattleTots agent
ecology performance. Compare results here against the integrated runs produced by
`scripts/run_with_tattletots.py`.
