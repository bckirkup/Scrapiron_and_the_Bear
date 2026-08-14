#!/usr/bin/env python3
"""Run FireEcology integrated with an orchestration layer (default: TattleTots).

Thin wrapper around fire_ecology.runner — prefer `fire-ecology sim --layer tattletots`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fire_ecology.runner import FireDomainHooks, run_fire_simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_with_tattletots",
        description="FireEcology with TattleTots agent ecology layer",
    )
    parser.add_argument("--config", type=Path, help="JSON config (domain + simulation sections)")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--grid-rows", type=int, default=20)
    parser.add_argument("--grid-cols", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    hooks = FireDomainHooks()
    cli_overrides: dict = {
        "layer": "tattletots",
        "verbose": args.verbose,
        "domain": {
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "seed": args.seed,
            "steps": args.steps,
        },
        "simulation": {
            "initial_population": args.population,
            "max_steps": args.steps,
            "seed": args.seed,
        },
    }
    if args.output:
        cli_overrides["output"] = str(args.output)

    run = hooks.load_run_context(
        config_path=str(args.config) if args.config else None,
        cli_overrides=cli_overrides,
    )
    run_fire_simulation(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
