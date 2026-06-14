#!/usr/bin/env python3
"""Run FireEcology (Scrapiron_and_the_Bear) simulation integrated with the TattleTots engine.

This script plugs the FireEcology domain adapter into the full TattleTots
agent ecology — agents compress sensor streams, evolve, form trophic
hierarchies, and escalate anomalies to human users.

Usage:
    python scripts/run_with_tattletots.py --config configs/tattletots_integration.json --output results.json
    python scripts/run_with_tattletots.py --steps 200 --seed 7 --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tattletots.engine.config import SimulationConfig
from tattletots.engine.world import World
from tattletots.output_schema import (
    CostMetrics,
    EcologyMetrics,
    RunSummary,
    SimulationOutput,
    TimeSeries,
)
from tattletots.telemetry.cost_accounting import CostAccumulator

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.metrics.fire_metrics import FireMetrics, StepMetrics


def main(argv: list[str] | None = None) -> int:
    """Run integrated FireEcology + TattleTots simulation."""
    parser = argparse.ArgumentParser(
        prog="run_with_tattletots",
        description="FireEcology: wildfire management integrated with TattleTots agent ecology",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file (contains 'simulation' and 'domain' sections)",
    )
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--population", type=int, default=20, help="Initial agent population")
    parser.add_argument("--grid-rows", type=int, default=20, help="Grid rows")
    parser.add_argument("--grid-cols", type=int, default=20, help="Grid columns")
    parser.add_argument("--output", type=Path, help="Path to write unified JSON results")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step progress")
    args = parser.parse_args(argv)

    # Load configuration
    if args.config:
        with open(args.config) as f:
            raw = json.load(f)
        sim_config = SimulationConfig(**raw.get("simulation", {}))
        domain_cfg = raw.get("domain", {})
        grid_rows = domain_cfg.get("grid_rows", 20)
        grid_cols = domain_cfg.get("grid_cols", 20)
        steps = domain_cfg.get("steps", args.steps)
        seed = domain_cfg.get("seed", args.seed)
        max_thermal_dim = domain_cfg.get("max_thermal_dim", None)
    else:
        sim_config = SimulationConfig(
            initial_population=args.population,
            max_steps=args.steps,
            seed=args.seed,
        )
        grid_rows = args.grid_rows
        grid_cols = args.grid_cols
        steps = args.steps
        seed = args.seed
        max_thermal_dim = None
        domain_cfg = {
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "steps": steps,
            "seed": seed,
        }

    # Build domain adapter
    adapter_kwargs: dict[str, object] = {
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "seed": seed,
    }
    if max_thermal_dim is not None:
        adapter_kwargs["max_thermal_dim"] = max_thermal_dim
    adapter = FireEcologyAdapter(**adapter_kwargs)  # type: ignore[arg-type]

    # Build TattleTots world
    world = World(config=sim_config)
    for stream in adapter.get_streams():
        world.add_stream(stream)
    for user in adapter.get_users():
        world.add_user(user)
    world.seed_population()

    # Run simulation
    cost_accumulator = CostAccumulator()
    fire_metrics = FireMetrics()
    start_time = time.time()

    print("=== FireEcology + TattleTots Integration ===")
    print(
        f"  Steps: {steps}, Grid: {grid_rows}x{grid_cols}, "
        f"Population: {sim_config.initial_population}, Seed: {seed}"
    )
    print()

    for step in range(steps):
        # Advance domain
        adapter.step(step)
        world.set_ground_truth(adapter.get_ground_truth(step))

        # Advance agent ecology
        record = world.step()

        if args.verbose and step % 50 == 0:
            active = adapter.fire_grid.active_fire_cells()
            print(
                f"  Step {step:4d}: pop={record.population:3d} "
                f"fires={len(active)} "
                f"reports={record.reports_issued} "
                f"trophic={record.max_trophic_level:.1f}"
            )

        # Cost accounting
        cost_dict = adapter.compute_costs(
            n_escalations=record.reports_issued,
            n_correct=record.correct_reports,
            n_false_alarms=record.false_alarms,
            n_missed=record.missed_events,
        )
        cost_accumulator.record_from_dict(record.time_step, cost_dict)

        # Fire domain metrics
        active = adapter.fire_grid.active_fire_cells()
        burned = adapter.fire_grid.burned_area()
        fire_metrics.record_detections_from_grid(
            [(r, c) for r, c in active], adapter.fire_grid, step
        )
        fire_metrics.record_step(
            StepMetrics(
                time_step=step,
                active_fires=len(active),
                burned_area=burned,
                surveillance_cost=cost_dict["surveillance_cost"],
                response_cost=cost_dict["response_cost"],
                damage_cost=cost_dict["damage_cost"],
            )
        )

        if record.population == 0:
            print("  ** Total extinction **")
            break

    wall_time = time.time() - start_time

    # Gather results
    summary = world.telemetry.summary()
    cost_summary = cost_accumulator.summary()

    print()
    print("=== Simulation Complete ===")
    print(f"  Final population: {summary['final_population']}")
    print(f"  Precision:        {summary['precision']:.2%}")
    print(f"  Total cost:       {cost_summary['total_cost']:.2f}")
    print(f"  Detection latency:{fire_metrics.mean_detection_latency:.1f}")
    print(f"  Wall time:        {wall_time:.1f}s")

    # Build unified output
    output = SimulationOutput(
        run_summary=RunSummary(
            domain="fire_ecology",
            steps_completed=world.telemetry.total_steps,
            seed=seed,
            wall_time_seconds=wall_time,
        ),
        simulation_config=sim_config.model_dump(),
        domain_config=domain_cfg,
        ecology_metrics=EcologyMetrics(
            final_population=int(summary["final_population"]),
            peak_population=int(summary["peak_population"]),
            total_births=int(summary["total_births"]),
            total_deaths=int(summary["total_deaths"]),
            total_reports=int(summary["total_reports"]),
            precision=float(summary["precision"]),
            max_trophic_depth=float(summary["max_trophic_depth"]),
            reached_equilibrium=bool(summary["reached_equilibrium"]),
        ),
        cost_metrics=CostMetrics(
            total_surveillance_cost=cost_summary["total_surveillance_cost"],
            total_response_cost=cost_summary["total_response_cost"],
            total_damage_cost=cost_summary["total_damage_cost"],
            total_cost=cost_summary["total_cost"],
            mean_cost_per_step=cost_summary["mean_cost_per_step"],
        ),
        domain_metrics=fire_metrics.summary(),
        time_series=TimeSeries(
            population=world.telemetry.population_history(),
            cost_per_step=cost_accumulator.cost_history(),
        ),
    )

    if args.output:
        output.write_json(args.output)
        print(f"\n  Results written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
