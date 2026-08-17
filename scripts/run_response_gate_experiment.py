"""Measure lever 5, the correctness-keyed response gate, on the wildfire instrument.

Runs the ordinary (evolved) arm at each ``reproduction_correctness_weight`` plus the
designed reporter at the largest weight, over one or more independent seed blocks, on
the agent-only OPIR-ablated path, and writes the JSON results and the Markdown writeup.

Usage:
    uv run --no-sync --no-build python scripts/run_response_gate_experiment.py \
        --steps 600 --jobs 8
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from fire_ecology.measurement.designed_reporter import DesignedReporterSpec, PayoffLevers
from fire_ecology.measurement.response_gate import (
    DEFAULT_WEIGHTS,
    HOLDOUT_SEEDS,
    PRIMARY_SEEDS,
    GateArm,
    assemble_block,
    assemble_results,
    gate_arms,
    markdown_report,
    results_json,
    run_arm_seed,
    seed_metrics,
)

_DOCS_DIR = Path("docs")
_REPORT_NAME = "response_gate_measurement.md"
_RESULTS_NAME = "response_gate_measurement.json"

_Task = tuple[DesignedReporterSpec, GateArm, int, str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--primary-seeds", type=int, nargs="+", default=list(PRIMARY_SEEDS))
    parser.add_argument("--holdout-seeds", type=int, nargs="+", default=list(HOLDOUT_SEEDS))
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_WEIGHTS),
        help="Swept reproduction_correctness_weight values; 0.0 is the reserves-only control.",
    )
    parser.add_argument("--grid-rows", type=int, default=20)
    parser.add_argument("--grid-cols", type=int, default=20)
    parser.add_argument("--n-cameras", type=int, default=3)
    parser.add_argument("--opir-cadence", type=int, default=5)
    parser.add_argument("--base-ignition-rate", type=float, default=0.0001)
    parser.add_argument("--initial-population", type=int, default=20)
    parser.add_argument("--max-population", type=int, default=60)
    parser.add_argument("--correct-report-attention-value", type=float, default=8.0)
    parser.add_argument("--break-even-precision", type=float, default=0.2)
    parser.add_argument(
        "--threshold-range", type=float, nargs=2, default=[0.05, 0.3], metavar=("LOW", "HIGH")
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--docs-dir", type=Path, default=_DOCS_DIR)
    return parser.parse_args(argv)


def _spec_from_args(args: argparse.Namespace) -> DesignedReporterSpec:
    low, high = (float(value) for value in args.threshold_range)
    levers = PayoffLevers(
        enabled=True,
        correct_report_attention_value=float(args.correct_report_attention_value),
        false_alarm_break_even_precision=float(args.break_even_precision),
        escalation_threshold_range=(low, high),
    )
    return DesignedReporterSpec(
        steps=args.steps,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        n_cameras=args.n_cameras,
        opir_cadence=args.opir_cadence,
        base_ignition_rate=args.base_ignition_rate,
        initial_population=args.initial_population,
        max_population=args.max_population,
        levers=levers,
    )


def _seed_blocks(args: argparse.Namespace) -> dict[str, list[int]]:
    blocks = {"primary": [int(seed) for seed in args.primary_seeds]}
    holdout = [int(seed) for seed in args.holdout_seeds]
    if holdout:
        blocks["holdout"] = holdout
    return blocks


def _run_task(task: _Task) -> dict[str, Any]:
    spec, arm, seed, _block = task
    return seed_metrics(arm, run_arm_seed(spec, arm, seed))


def _tasks(
    spec: DesignedReporterSpec,
    arms: tuple[GateArm, ...],
    blocks: dict[str, list[int]],
) -> list[_Task]:
    return [
        (spec, arm, seed, block)
        for block, seeds in blocks.items()
        for arm in arms
        for seed in seeds
    ]


def _completed_metrics(tasks: list[_Task], jobs: int) -> list[dict[str, Any]]:
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(_run_task, tasks))
    metrics: list[dict[str, Any]] = []
    for task in tasks:
        print(f"running block {task[3]} arm {task[1].name} seed {task[2]} ...", flush=True)
        metrics.append(_run_task(task))
    return metrics


def _run_blocks(
    spec: DesignedReporterSpec,
    arms: tuple[GateArm, ...],
    blocks: dict[str, list[int]],
    jobs: int,
) -> dict[str, dict[str, Any]]:
    tasks = _tasks(spec, arms, blocks)
    per_block: dict[str, dict[str, list[dict[str, Any]]]] = {
        block: {arm.name: [] for arm in arms} for block in blocks
    }
    for task, metrics in zip(tasks, _completed_metrics(tasks, jobs), strict=True):
        per_block[task[3]][task[1].name].append(metrics)
    return {block: assemble_block(arms, seeds, per_block[block]) for block, seeds in blocks.items()}


def _reproduce_command(args: argparse.Namespace) -> str:
    weights = " ".join(f"{float(weight):g}" for weight in args.weights)
    return (
        "uv run --no-sync --no-build python scripts/run_response_gate_experiment.py \\\n"
        f"    --steps {args.steps} --weights {weights} \\\n"
        f"    --primary-seeds {' '.join(str(int(s)) for s in args.primary_seeds)} \\\n"
        f"    --holdout-seeds {' '.join(str(int(s)) for s in args.holdout_seeds)}"
    )


def _artifact_path(output_dir: Path, name: str) -> Path:
    """Resolve one artifact inside ``output_dir``, rejecting escapes from that directory."""
    path = (output_dir / name).resolve()
    if path.parent != output_dir:
        raise ValueError(f"artifact {name} would be written outside {output_dir}")
    return path


def _write_artifacts(results: dict[str, Any], docs_dir: Path, command: str) -> list[Path]:
    output_dir = docs_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = _artifact_path(output_dir, _RESULTS_NAME)
    results_path.write_text(
        json.dumps(results_json(results), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = _artifact_path(output_dir, _REPORT_NAME)
    report_path.write_text(markdown_report(results, command), encoding="utf-8")
    return [results_path, report_path]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = _spec_from_args(args)
    arms = gate_arms([float(weight) for weight in args.weights])
    blocks = _run_blocks(spec, arms, _seed_blocks(args), max(int(args.jobs), 1))
    results = assemble_results(spec, arms, blocks)

    for path in _write_artifacts(results, args.docs_dir, _reproduce_command(args)):
        print(f"wrote {path}")
    print(json.dumps(results["verdicts"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
