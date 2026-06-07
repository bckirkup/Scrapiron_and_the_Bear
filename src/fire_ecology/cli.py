"""CLI entrypoint for fire-ecology simulation."""

from __future__ import annotations

import argparse
import json

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.comparison import (
    ComparisonConfig,
    format_comparison_json,
    format_comparison_table,
    run_comparison,
)
from fire_ecology.metrics.fire_metrics import FireMetrics, StepMetrics


def _run_simulation(args: argparse.Namespace) -> None:
    """Run a single fire ecology simulation with metrics."""
    adapter = FireEcologyAdapter(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        seed=args.seed,
        max_thermal_dim=args.max_thermal_dim,
    )
    metrics = FireMetrics()

    for step in range(args.steps):
        adapter.step(step)
        active = adapter.fire_grid.active_fire_cells()
        burned = adapter.fire_grid.burned_area()

        # Track detection latencies from newly burning cells
        metrics.record_detections_from_grid([(r, c) for r, c in active], adapter.fire_grid, step)

        # Compute domain costs via the adapter
        n_active = len(active)
        costs = adapter.compute_costs(
            n_escalations=0,
            n_correct=0,
            n_false_alarms=0,
            n_missed=n_active,
        )

        step_metrics = StepMetrics(
            time_step=step,
            active_fires=n_active,
            burned_area=burned,
            surveillance_cost=costs["surveillance_cost"],
            response_cost=costs["response_cost"],
            damage_cost=costs["damage_cost"],
        )
        metrics.record_step(step_metrics)

        if args.verbose and n_active > 0:
            print(
                f"Step {step:4d}: active={n_active:3d}  burned={burned:4d}  "
                f"weather_fwi={adapter.weather.fire_weather_index:.2f}"
            )

    summary = metrics.summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n=== Fire Ecology Simulation Summary ===")
        for key, val in summary.items():
            print(f"  {key}: {val:.4f}")


def _run_compare(args: argparse.Namespace) -> None:
    """Run head-to-head architecture comparison."""
    config = ComparisonConfig(
        steps=args.steps,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        seed=args.seed,
        n_drones=args.n_drones,
        include_a4=not args.no_a4,
        max_thermal_dim=args.max_thermal_dim,
    )
    results = run_comparison(config)

    if args.json:
        print(format_comparison_json(results))
    else:
        print("\n=== Head-to-Head Architecture Comparison ===")
        print(
            f"    Steps: {config.steps}  Grid: {config.grid_rows}x{config.grid_cols}"
            f"  Seed: {config.seed}  Drones: {config.n_drones}\n"
        )
        print(format_comparison_table(results))
        print()


def main(argv: list[str] | None = None) -> None:
    """Run a fire ecology simulation or architecture comparison."""
    parser = argparse.ArgumentParser(
        description="FireEcology: autonomous fire-regime management simulation"
    )
    # Default to "sim" when no subcommand is given (backward compat)
    subparsers = parser.add_subparsers(dest="command", required=False)

    # --- sim subcommand (default behavior) ---
    sim_parser = subparsers.add_parser("sim", help="Run a single simulation")
    _add_common_args(sim_parser)

    # --- compare subcommand ---
    cmp_parser = subparsers.add_parser("compare", help="Head-to-head architecture comparison")
    _add_common_args(cmp_parser)
    cmp_parser.add_argument("--n-drones", type=int, default=10, help="Drones per architecture")
    cmp_parser.add_argument("--no-a4", action="store_true", help="Exclude A4/BMA from comparison")

    # Pre-parse: if first arg isn't a known subcommand, inject "sim"
    effective_argv = argv if argv is not None else []
    if effective_argv and effective_argv[0] not in ("sim", "compare", "-h", "--help"):
        effective_argv = ["sim", *effective_argv]
    elif not effective_argv:
        effective_argv = ["sim"]

    args = parser.parse_args(effective_argv)

    if args.command == "compare":
        _run_compare(args)
    else:
        _run_simulation(args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--grid-rows", type=int, default=20, help="Grid rows")
    parser.add_argument("--grid-cols", type=int, default=20, help="Grid columns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print per-step output")
    parser.add_argument("--json", action="store_true", help="Output metrics as JSON")
    parser.add_argument(
        "--max-thermal-dim",
        type=int,
        default=None,
        help=(
            "Maximum dimensionality for the thermal detection stream. "
            f"Defaults to {FireEcologyAdapter.DEFAULT_THERMAL_DIM}. "
            "Use grid_rows*grid_cols for full resolution."
        ),
    )


if __name__ == "__main__":
    main()
