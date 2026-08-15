"""Measure agent-only fire detection versus grounded raw-stream access.

Writes one ``tattletots.output_schema.SimulationOutput`` JSON per arm and seed,
plus a flat summary JSON with per-arm means, so the cross-domain comparison can
read the same schema every other domain emits.

Usage:
    uv run --no-sync --no-build python scripts/measure_grounded_access.py \
        --steps 200 --seeds 42 43 44 45 46 --output-dir docs/grounded_access
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fire_ecology.measurement.grounded_access import (
    DEFAULT_GROUNDED_FRACTIONS,
    ArmSpec,
    SweepSpec,
    instrument_nulls,
    run_arm,
    sweep_arms,
)

_MEAN_KEYS = (
    "agent_only_detection_rate",
    "agent_only_step_detection_rate",
    "agent_only_cell_recall",
    "reported_detection_rate",
    "report_precision",
    "report_false_alarm_rate",
    "null_chance_precision",
    "null_static_prior_precision",
    "attention_solvent_fraction",
    "mean_per_capita_attention_capacity",
    "grounded_yield_share",
    "effective_grounded_yield_share",
    "parent_child_reproductive_correlation",
    "final_population",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--grid-rows", type=int, default=20)
    parser.add_argument("--grid-cols", type=int, default=20)
    parser.add_argument("--n-drones", type=int, default=10)
    parser.add_argument("--instrument-steps", type=int, default=200)
    parser.add_argument(
        "--grounded-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_GROUNDED_FRACTIONS),
        help="grounded_input_fraction values to measure at identical seeds",
    )
    parser.add_argument(
        "--no-assisted-arms",
        action="store_true",
        help="Measure only the OPIR-ablated arms, skipping their OPIR-assisted twins",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("docs/grounded_access"))
    return parser.parse_args(argv)


def _relative(path: Path) -> str:
    cwd = Path.cwd()
    return str(path.relative_to(cwd)) if path.is_relative_to(cwd) else str(path)


def _mean_or_none(values: list[Any]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)

    aggregated: list[dict[str, Any]] = []
    for label, arm_rows in grouped.items():
        first = arm_rows[0]
        entry: dict[str, Any] = {
            "label": label,
            "grounded_input_fraction": first["grounded_input_fraction"],
            "opir_backstop_ablated": first["opir_backstop_ablated"],
            "n_seeds": len(arm_rows),
            "seeds": [row["seed"] for row in arm_rows],
            "extinct_seeds": [row["seed"] for row in arm_rows if row["ecology_extinct"]],
        }
        for key in _MEAN_KEYS:
            entry[f"mean_{key}"] = _mean_or_none([row[key] for row in arm_rows])
        aggregated.append(entry)
    return aggregated


def _run_one(
    spec: ArmSpec,
    sweep: SweepSpec,
    nulls: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    print(f"running arm {spec.label} seed {sweep.seed} ...", flush=True)
    result = run_arm(spec, sweep, nulls=nulls)
    path = output_dir / f"{spec.label}_seed{sweep.seed}.json"
    result.output.write_json(path)
    row = result.key_numbers()
    row["seed"] = sweep.seed
    row["output_json"] = _relative(path)
    print(json.dumps(row, indent=1, sort_keys=True), flush=True)
    return row


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    specs = sweep_arms(args.grounded_fractions, include_assisted=not args.no_assisted_arms)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    nulls_by_seed: dict[int, dict[str, Any]] = {}
    for seed in args.seeds:
        sweep = SweepSpec(
            steps=args.steps,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            seed=seed,
            n_drones=args.n_drones,
            instrument_steps=args.instrument_steps,
        )
        nulls = instrument_nulls(sweep)
        nulls_by_seed[seed] = nulls
        rows.extend(_run_one(spec, sweep, nulls, output_dir) for spec in specs)

    summary = {
        "sweep": {
            "steps": args.steps,
            "seeds": list(args.seeds),
            "grid_rows": args.grid_rows,
            "grid_cols": args.grid_cols,
            "n_drones": args.n_drones,
            "instrument_steps": args.instrument_steps,
            "grounded_fractions": list(args.grounded_fractions),
        },
        "instrument_nulls_by_seed": {str(k): v for k, v in nulls_by_seed.items()},
        "runs": rows,
        "arm_means": _aggregate(rows),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"wrote {_relative(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
