"""CLI entrypoint for fire-ecology simulation."""

from __future__ import annotations

import argparse
import json

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.metrics.fire_metrics import FireMetrics, StepMetrics


def main(argv: list[str] | None = None) -> None:
    """Run a fire ecology simulation."""
    parser = argparse.ArgumentParser(
        description="FireEcology: autonomous fire-regime management simulation"
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--grid-rows", type=int, default=20, help="Grid rows")
    parser.add_argument("--grid-cols", type=int, default=20, help="Grid columns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print per-step output")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")
    args = parser.parse_args(argv)

    adapter = FireEcologyAdapter(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        seed=args.seed,
    )
    metrics = FireMetrics()

    for step in range(args.steps):
        adapter.step(step)
        active = adapter.fire_grid.active_fire_cells()
        burned = adapter.fire_grid.burned_area()

        step_metrics = StepMetrics(
            time_step=step,
            active_fires=len(active),
            burned_area=burned,
        )
        metrics.record_step(step_metrics)

        if args.verbose and len(active) > 0:
            print(
                f"Step {step:4d}: active={len(active):3d}  burned={burned:4d}  "
                f"weather_fwi={adapter.weather.fire_weather_index:.2f}"
            )

    summary = metrics.summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n=== Fire Ecology Simulation Summary ===")
        for key, val in summary.items():
            print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
