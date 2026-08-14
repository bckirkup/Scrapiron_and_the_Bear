#!/usr/bin/env python3
"""Parameter Scan Runner for Fire Ecology Baselines (Without TattleTots).

Run from the workspace root (parent of all repos):

    python Scrapiron_and_the_Bear/baselines/run_fire_ecology_baselines.py --smoke-test
    python Scrapiron_and_the_Bear/baselines/run_fire_ecology_baselines.py --workers 8
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fire_ecology.comparison import ComparisonConfig, run_comparison

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _parent in [_SCRIPT_DIR, *_SCRIPT_DIR.parents]:
    _large_experiments = _parent / "TattleTots" / "Large Experiments"
    if (_large_experiments / "baseline_parallel.py").is_file():
        sys.path.insert(0, str(_large_experiments))
        break
else:
    sys.exit(
        "[-] Error: Could not find TattleTots/Large Experiments/baseline_parallel.py.\n"
        "    Ensure all repos are cloned as siblings under a common workspace root."
    )


def _safe_path_under_base(raw: Path, base: Path) -> Path:
    resolved = raw.resolve()
    base_resolved = base.resolve()
    if not resolved.is_relative_to(base_resolved):
        raise ValueError(f"Path escapes allowed directory: {raw}")
    return resolved


def _safe_config_path(config: Path) -> Path:
    return _safe_path_under_base(config, _REPO_ROOT)


def _safe_output_dir(name: str) -> Path:
    return _safe_path_under_base(_REPO_ROOT / name, _REPO_ROOT)


def run_single_simulation(
    steps: int,
    seed: int,
    n_cameras: int,
    n_drones: int,
    grid_rows: int,
    grid_cols: int,
    base_ignition_rate: float,
    weather_volatility: float,
) -> dict[str, Any]:
    """Run a single fire ecology baseline comparison (A0-A3)."""
    start_time = time.time()

    config = ComparisonConfig(
        steps=steps,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        seed=seed,
        n_drones=n_drones,
        n_cameras=n_cameras,
        base_ignition_rate=base_ignition_rate,
        weather_volatility=weather_volatility,
        include_a4=False,
    )
    baselines = run_comparison(config)

    elapsed_time = time.time() - start_time
    baseline_results = {b.name: asdict(b) for b in baselines}

    return {
        "status": "success",
        "elapsed_seconds": elapsed_time,
        "config": {
            "steps": steps,
            "seed": seed,
            "n_cameras": n_cameras,
            "n_drones": n_drones,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "base_ignition_rate": base_ignition_rate,
            "weather_volatility": weather_volatility,
        },
        "baselines": baseline_results,
    }


def _load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return json.load(f)


def _build_factor_grid(config_data: dict[str, Any], smoke_test: bool) -> dict[str, list[Any]]:
    if smoke_test:
        return {
            "deployment_phase": ["phase_2"],
            "sensor_dropout": ["0%"],
            "drone_fleet_size": [10],
            "ignition_rate": ["medium"],
            "weather_volatility": ["medium"],
        }
    factors = config_data.get("factors", {})
    return {
        "deployment_phase": factors.get("deployment_phase", ["phase_2"]),
        "sensor_dropout": factors.get("sensor_dropout", ["0%"]),
        "drone_fleet_size": factors.get("drone_fleet_size", [10]),
        "ignition_rate": factors.get("ignition_rate", ["medium"]),
        "weather_volatility": factors.get("weather_volatility", ["medium"]),
    }


def _build_runs_to_execute(
    config_data: dict[str, Any],
    *,
    steps: int,
    seeds: list[int],
    grid_rows: int,
    grid_cols: int,
    factor_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    phase_defs = config_data.get("phases", {})
    ignition_map = config_data.get("ignition_rates", {"medium": 0.0001})
    volatility_map = config_data.get("weather_volatility", {"medium": 1.0})
    factor_names = list(factor_grid.keys())
    factor_values = [factor_grid[name] for name in factor_names]

    runs_to_execute: list[dict[str, Any]] = []
    for combo in itertools.product(*factor_values):
        combo_dict = dict(zip(factor_names, combo, strict=True))
        phase = combo_dict["deployment_phase"]
        dropout = combo_dict["sensor_dropout"]
        n_drones = int(combo_dict["drone_fleet_size"])
        ignition_level = combo_dict["ignition_rate"]
        volatility_level = combo_dict["weather_volatility"]

        dropout_frac = float(str(dropout).replace("%", "")) / 100.0
        base_cameras = phase_defs.get(phase, {}).get("n_cameras", 3)
        n_cameras = max(1, int(base_cameras * (1.0 - dropout_frac)))
        base_ignition_rate = float(ignition_map[ignition_level])
        weather_volatility = float(volatility_map[volatility_level])

        for seed in seeds:
            drop_tag = str(dropout).replace("%", "")
            run_name = (
                f"fe_baselines_{phase}_drop{drop_tag}_d{n_drones}"
                f"_ign{ignition_level}_wx{volatility_level}_s{seed}"
            )
            runs_to_execute.append(
                {
                    "name": run_name,
                    "steps": steps,
                    "seed": seed,
                    "n_cameras": n_cameras,
                    "n_drones": n_drones,
                    "grid_rows": grid_rows,
                    "grid_cols": grid_cols,
                    "base_ignition_rate": base_ignition_rate,
                    "weather_volatility": weather_volatility,
                    "metadata": {
                        "deployment_phase": phase,
                        "sensor_dropout": dropout,
                        "drone_fleet_size": n_drones,
                        "ignition_rate": ignition_level,
                        "weather_volatility": volatility_level,
                        "n_cameras": n_cameras,
                    },
                }
            )
    return runs_to_execute


def _baseline_summary_row(name: str, run_res: dict[str, Any]) -> tuple[str, str, str, str, str]:
    if run_res.get("status") == "success":
        status = "success"
        elapsed = f"{run_res.get('elapsed_seconds', 0.0):.1f}"
        a2 = run_res["baselines_summary"].get("A2 Centralized", {})
        burned = str(a2.get("burned_cells", "N/A"))
        cost = f"{a2.get('cost', 0.0):,.1f}" if a2 else "N/A"
        return name, status, elapsed, burned, cost
    return name, "failed", "N/A", "N/A", "N/A"


def _print_run_summary(results_key: dict[str, Any]) -> None:
    print("\n=== Fire Ecology Baselines Parameter Scan Summary ===")
    print(
        f"{'Run Name':<55} | {'Status':<10} | {'Time (s)':<8} | {'A2 Burned':<10} | {'A2 Cost':<10}"
    )
    print("-" * 105)
    for name, run_res in results_key["runs"].items():
        row = _baseline_summary_row(name, run_res)
        print(f"{row[0]:<55} | {row[1]:<10} | {row[2]:<8} | {row[3]:<10} | {row[4]:<10}")
    print("=" * 105)


def _write_output_files(
    output_dir: Path,
    results_key: dict[str, Any],
    all_results: dict[str, Any],
    logs: list[str],
    *,
    n_runs: int,
    total_elapsed: float,
) -> None:
    key_file_path = output_dir / "key.json"
    with open(key_file_path, "w") as f:
        json.dump(results_key, f, indent=2)
    print(f"[+] Parameter scan summary key written to: {key_file_path}")

    results_file_path = output_dir / "results.json"
    with open(results_file_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[+] Consolidated results written to: {results_file_path}")

    log_file_path = output_dir / "all_runs.log"
    with open(log_file_path, "w") as f:
        f.write("=== Fire Ecology Baselines Parameter Scan Log ===\n")
        f.write(f"Timestamp: {datetime.datetime.now(datetime.UTC).isoformat()}\n")
        f.write(f"Total Runs: {n_runs}\n")
        f.write(f"Total Elapsed Time: {total_elapsed:.1f}s\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(logs))
    print(f"[+] Consolidated logs written to: {log_file_path}")


def _execute_runs(
    args: argparse.Namespace,
    runs_to_execute: list[dict[str, Any]],
    worker_count: int,
    results_key: dict[str, Any],
    all_results: dict[str, Any],
    logs: list[str],
) -> None:
    from baseline_parallel import run_process_pool

    def _store_success(run: dict[str, Any], res: dict[str, Any]) -> None:
        name = run["name"]
        results_key["runs"][name] = {
            "status": res["status"],
            "elapsed_seconds": res["elapsed_seconds"],
            "metadata": run["metadata"],
            "baselines_summary": {
                b_name: {
                    "detections": b_data["detections"],
                    "suppressions": b_data["suppressions"],
                    "burned_cells": b_data["burned_cells"],
                    "cost": b_data["cost"],
                }
                for b_name, b_data in res["baselines"].items()
            },
        }
        all_results[name] = res.copy()
        logs.append(f"[+] Completed: {name} in {res['elapsed_seconds']:.2f}s")

    def _store_failure(run: dict[str, Any], exc: Exception) -> None:
        results_key["runs"][run["name"]] = {"status": "failed", "error": str(exc)}

    submit_kwargs = [
        (
            run["steps"],
            run["seed"],
            run["n_cameras"],
            run["n_drones"],
            run["grid_rows"],
            run["grid_cols"],
            run["base_ignition_rate"],
            run["weather_volatility"],
        )
        for run in runs_to_execute
    ]

    if args.parallel:
        run_process_pool(
            run_single_simulation,
            submit_kwargs,
            runs_to_execute,
            max_workers=worker_count,
            on_success=_store_success,
            on_failure=_store_failure,
        )
        return

    for run, kwargs in zip(runs_to_execute, submit_kwargs, strict=True):
        name = run["name"]
        try:
            _store_success(run, run_single_simulation(*kwargs))
            print(f"[+] Completed: {name}")
        except Exception as e:
            _store_failure(run, e)
            print(f"[-] Run '{name}' failed: {e}")


def main() -> int:
    from baseline_parallel import resolve_worker_count

    parser = argparse.ArgumentParser(description="Parameter Scan Runner for Fire Ecology Baselines")
    parser.add_argument(
        "--config",
        type=Path,
        default=_SCRIPT_DIR / "fire_ecology_baselines_config.json",
        help="Path to parameter scan config JSON file",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a fast smoke test")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--no-parallel", action="store_false", dest="parallel")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: min(CPU count, job count))",
    )
    args = parser.parse_args()

    try:
        config_path = _safe_config_path(args.config)
    except ValueError as exc:
        print(f"[-] Error: {exc}")
        return 1

    if not config_path.exists():
        print(f"[-] Error: Config file not found at {config_path}")
        return 1

    config_data = _load_config(config_path)

    output_dir_name = (
        "fire_ecology_baselines_smoke_results"
        if args.smoke_test
        else config_data.get("output_directory", "fire_ecology_baselines_results")
    )
    try:
        output_dir = _safe_output_dir(output_dir_name)
    except ValueError as exc:
        print(f"[-] Error: {exc}")
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = 5 if args.smoke_test else config_data.get("steps", 800)
    seeds = [42] if args.smoke_test else config_data.get("seeds", [42, 43, 44])
    grid_rows = config_data.get("grid_rows", 20)
    grid_cols = config_data.get("grid_cols", 20)
    factor_grid = _build_factor_grid(config_data, args.smoke_test)
    runs_to_execute = _build_runs_to_execute(
        config_data,
        steps=steps,
        seeds=seeds,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        factor_grid=factor_grid,
    )

    n_jobs = len(runs_to_execute)
    worker_count = resolve_worker_count(args.workers, n_jobs)

    print(f"[*] Results will be saved to: {output_dir}")
    print(f"[*] Generated {n_jobs} total run configurations.")
    if args.parallel:
        print(
            f"[*] Execution mode: PARALLEL (ProcessPoolExecutor, "
            f"{worker_count} worker process{'es' if worker_count != 1 else ''}, "
            f"PID {os.getpid()} parent)"
        )
    else:
        print(f"[*] Execution mode: SEQUENTIAL (single process, PID {os.getpid()})")
    print("=" * 60)

    results_key: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "is_smoke_test": args.smoke_test,
        "output_directory": str(output_dir),
        "runs": {},
    }

    start_time = time.time()
    all_results: dict[str, Any] = {}
    logs: list[str] = []
    _execute_runs(args, runs_to_execute, worker_count, results_key, all_results, logs)

    total_elapsed = time.time() - start_time
    print("=" * 60)
    print(f"[+] All runs finished in {total_elapsed:.1f}s.")

    _write_output_files(
        output_dir,
        results_key,
        all_results,
        logs,
        n_runs=len(runs_to_execute),
        total_elapsed=total_elapsed,
    )
    _print_run_summary(results_key)

    any_failed = any(r.get("status") == "failed" for r in results_key["runs"].values())
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
