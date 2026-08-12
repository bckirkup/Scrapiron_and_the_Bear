"""CLI entrypoint for fire-ecology simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fire_ecology.comparison import (
    ComparisonConfig,
    format_comparison_json,
    format_comparison_table,
    run_comparison,
)
from fire_ecology.runner import FireDomainHooks, run_fire_batch, run_fire_simulation


def _run_simulation(args: argparse.Namespace) -> None:
    hooks = FireDomainHooks()
    run = hooks.load_run_context(
        config_path=args.config,
        cli_overrides={
            "domain": {
                "grid_rows": args.grid_rows,
                "grid_cols": args.grid_cols,
                "seed": args.seed,
                "steps": args.steps,
                **({"max_thermal_dim": args.max_thermal_dim} if args.max_thermal_dim else {}),
            },
            "layer": args.layer,
            "verbose": args.verbose,
            "output": str(args.output) if args.output else None,
        },
    )
    result = run_fire_simulation(run)
    if args.json and not args.output:
        print(json.dumps(result.to_dict(), indent=2))


def _run_batch(args: argparse.Namespace) -> None:
    results = run_fire_batch(
        Path(args.config),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        parallel=args.parallel,
        workers=args.workers,
        verbose=args.verbose,
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n=== Batch Summary ===")
        for name, res in results["runs"].items():
            print(f"  {name}: {res.get('status')} ({res.get('layer')})")


def _run_compare(args: argparse.Namespace) -> None:
    config = ComparisonConfig(
        steps=args.steps,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        seed=args.seed,
        n_drones=args.n_drones,
        include_a4=not args.no_a4,
        include_a4_opir_ablation=args.a4_opir_ablation,
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
    parser = argparse.ArgumentParser(
        description="FireEcology: autonomous fire-regime management simulation"
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    sim_parser = subparsers.add_parser("sim", help="Run a single simulation")
    _add_common_args(sim_parser)
    sim_parser.add_argument(
        "--layer",
        default="domain_only",
        choices=["domain_only", "tattletots"],
        help="Orchestration layer (default: domain_only)",
    )
    sim_parser.add_argument("--config", type=str, help="JSON config file")
    sim_parser.add_argument("--output", type=Path, help="Write results JSON")

    batch_parser = subparsers.add_parser("batch", help="Run a batch of simulations")
    batch_parser.add_argument("--config", type=str, required=True, help="Batch JSON config")
    batch_parser.add_argument("--output-dir", type=Path, help="Output directory")
    batch_parser.add_argument("--parallel", action="store_true")
    batch_parser.add_argument("--workers", type=int, default=None)
    batch_parser.add_argument("--verbose", action="store_true")
    batch_parser.add_argument("--json", action="store_true")

    cmp_parser = subparsers.add_parser("compare", help="Head-to-head architecture comparison")
    _add_common_args(cmp_parser)
    cmp_parser.add_argument("--n-drones", type=int, default=10)
    cmp_parser.add_argument("--no-a4", action="store_true")
    cmp_parser.add_argument(
        "--a4-opir-ablation",
        action="store_true",
        help="Include a second A4 arm with the OPIR backstop disabled",
    )

    effective_argv = argv if argv is not None else []
    if effective_argv and effective_argv[0] not in ("sim", "batch", "compare", "-h", "--help"):
        effective_argv = ["sim", *effective_argv]
    elif not effective_argv:
        effective_argv = ["sim"]

    args = parser.parse_args(effective_argv)

    if args.command == "batch":
        _run_batch(args)
    elif args.command == "compare":
        _run_compare(args)
    else:
        _run_simulation(args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--grid-rows", type=int, default=20)
    parser.add_argument("--grid-cols", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-thermal-dim", type=int, default=None)


if __name__ == "__main__":
    main()
