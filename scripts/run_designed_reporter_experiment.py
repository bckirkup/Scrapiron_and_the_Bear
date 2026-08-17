"""Measure the wildfire domain's exploitable margin with a designed reporter.

Runs four policy arms — evolved (`ordinary`), `all_designed_seed`, `invasion`,
and the diagnostic `oracle_upper_bound` — over many seeds on the agent-only,
OPIR-ablated path, and writes the JSON results, one
``tattletots.output_schema.SimulationOutput`` per arm, and the Markdown writeup.

Usage:
    uv run --no-sync --no-build python scripts/run_designed_reporter_experiment.py \
        --steps 200 --seeds 42 43 44 --jobs 4
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from fire_ecology.measurement.designed_reporter import (
    DEFAULT_SEEDS,
    POLICY_ARMS,
    DesignedReporterSpec,
    PayoffLevers,
    SeedRun,
    assemble_results,
    markdown_report,
    results_json,
    run_seed,
    simulation_output,
)

_DOCS_DIR = Path("docs")
_REPORT_NAME = "designed_reporter_measurement.md"
_RESULTS_NAME = "designed_reporter_measurement.json"
_OUTPUT_SUBDIR = "designed_reporter"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--grid-rows", type=int, default=20)
    parser.add_argument("--grid-cols", type=int, default=20)
    parser.add_argument("--n-cameras", type=int, default=3)
    parser.add_argument("--opir-cadence", type=int, default=5)
    parser.add_argument("--base-ignition-rate", type=float, default=0.0001)
    parser.add_argument("--initial-population", type=int, default=20)
    parser.add_argument("--max-population", type=int, default=60)
    parser.add_argument("--grounded-input-fraction", type=float, default=0.67)
    parser.add_argument("--grounded-attractiveness-multiplier", type=float, default=1.0)
    parser.add_argument("--max-input-streams", type=int, default=3)
    parser.add_argument("--invasion-fraction", type=float, default=0.15)
    parser.add_argument(
        "--payoff-levers",
        action="store_true",
        help=(
            "Switch on the engine's measured payoff levers (verified-correctness "
            "attention income, merit-ordered reproduction, false-alarm pricing, "
            "score-unit escalation calibration). Off by default, which reproduces the "
            "committed numbers."
        ),
    )
    parser.add_argument(
        "--correct-report-attention-value",
        type=float,
        default=8.0,
        help="Attention value of a verified-correct report; used with --payoff-levers.",
    )
    parser.add_argument(
        "--break-even-precision",
        type=float,
        default=0.2,
        help="False-alarm pricing target precision; used with --payoff-levers.",
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        default=[0.05, 0.3],
        metavar=("LOW", "HIGH"),
        help="Starting escalation_threshold range; used with --payoff-levers.",
    )
    parser.add_argument(
        "--correctness-weight",
        type=float,
        default=0.0,
        help=(
            "Lever 5, the response gate: share of reproductive merit carried by rank in "
            "verified correctness. Used with --payoff-levers; 0.0 is the reserves-only "
            "control."
        ),
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--docs-dir", type=Path, default=_DOCS_DIR)
    return parser.parse_args(argv)


def _spec_from_args(args: argparse.Namespace) -> DesignedReporterSpec:
    return DesignedReporterSpec(
        steps=args.steps,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        n_cameras=args.n_cameras,
        opir_cadence=args.opir_cadence,
        base_ignition_rate=args.base_ignition_rate,
        initial_population=args.initial_population,
        max_population=args.max_population,
        grounded_input_fraction=args.grounded_input_fraction,
        grounded_attractiveness_multiplier=args.grounded_attractiveness_multiplier,
        max_input_streams=args.max_input_streams,
        invasion_fraction=args.invasion_fraction,
        levers=levers_from_args(args),
    )


def levers_from_args(args: argparse.Namespace) -> PayoffLevers:
    """Payoff-lever settings from the CLI; disabled unless ``--payoff-levers`` is given."""
    low, high = (float(value) for value in args.threshold_range)
    return PayoffLevers(
        enabled=bool(args.payoff_levers),
        correct_report_attention_value=float(args.correct_report_attention_value),
        false_alarm_break_even_precision=float(args.break_even_precision),
        escalation_threshold_range=(low, high),
        reproduction_correctness_weight=float(args.correctness_weight),
    )


def _run_task(task: tuple[DesignedReporterSpec, int, str]) -> SeedRun:
    spec, seed, arm = task
    return run_seed(spec, seed, arm)


def _run_all(
    spec: DesignedReporterSpec,
    seeds: list[int],
    jobs: int,
) -> dict[str, list[SeedRun]]:
    tasks = [(spec, seed, arm) for arm in POLICY_ARMS for seed in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            completed = list(pool.map(_run_task, tasks))
    else:
        completed = []
        for task in tasks:
            print(f"running arm {task[2]} seed {task[1]} ...", flush=True)
            completed.append(_run_task(task))
    runs: dict[str, list[SeedRun]] = {arm: [] for arm in POLICY_ARMS}
    for task, run in zip(tasks, completed, strict=True):
        runs[task[2]].append(run)
    return runs


def _write_artifacts(results: dict[str, Any], docs_dir: Path) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = docs_dir / _OUTPUT_SUBDIR
    outputs_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    results_path = docs_dir / _RESULTS_NAME
    results_path.write_text(
        json.dumps(results_json(results), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written.append(results_path)

    report_path = docs_dir / _REPORT_NAME
    report_path.write_text(markdown_report(results), encoding="utf-8")
    written.append(report_path)

    for arm in results["runs"]:
        path = outputs_dir / f"{arm}.json"
        simulation_output(results, arm).write_json(path)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = _spec_from_args(args)
    seeds = [int(seed) for seed in args.seeds]
    runs = _run_all(spec, seeds, max(int(args.jobs), 1))
    results = assemble_results(spec, seeds, runs)

    for path in _write_artifacts(results, args.docs_dir):
        print(f"wrote {path}")
    print(json.dumps(results["margin"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
