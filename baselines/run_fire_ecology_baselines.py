#!/usr/bin/env python3
"""Parameter Scan Runner for Fire Ecology Baselines (Without TattleTots).

This script runs a parameter scan for the Fire Ecology simulation using ONLY
the baseline management architectures (A0, A1, A2, A3). It sweeps deployment
phase, sensor dropout, drone fleet size, ignition rate, and weather volatility,
running each combination in triplicate for 800 steps.

All results are consolidated into exactly three output files to prevent clutter.

Usage:
    python run_fire_ecology_baselines.py --smoke-test
    python run_fire_ecology_baselines.py
"""

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fire_ecology.comparison import ComparisonConfig, run_comparison


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Parameter Scan Runner for Fire Ecology Baselines")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("fire_ecology_baselines_config.json"),
        help="Path to parameter scan config JSON file",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast smoke test of the parameter scan",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run simulations in parallel (default: True)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Run simulations sequentially",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"[-] Error: Config file not found at {args.config}")
        return 1

    with open(args.config) as f:
        config_data = json.load(f)

    output_dir_name = (
        "fire_ecology_baselines_smoke_results"
        if args.smoke_test
        else config_data.get("output_directory", "fire_ecology_baselines_results")
    )
    output_dir = Path(output_dir_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = 5 if args.smoke_test else config_data.get("steps", 800)
    seeds = [42] if args.smoke_test else config_data.get("seeds", [42, 43, 44])
    factors = config_data.get("factors", {})
    grid_rows = config_data.get("grid_rows", 20)
    grid_cols = config_data.get("grid_cols", 20)
    phase_defs = config_data.get("phases", {})
    ignition_map = config_data.get("ignition_rates", {"medium": 0.0001})
    volatility_map = config_data.get("weather_volatility", {"medium": 1.0})

    if args.smoke_test:
        factor_grid = {
            "deployment_phase": ["phase_2"],
            "sensor_dropout": ["0%"],
            "drone_fleet_size": [10],
            "ignition_rate": ["medium"],
            "weather_volatility": ["medium"],
        }
    else:
        factor_grid = {
            "deployment_phase": factors.get("deployment_phase", ["phase_2"]),
            "sensor_dropout": factors.get("sensor_dropout", ["0%"]),
            "drone_fleet_size": factors.get("drone_fleet_size", [10]),
            "ignition_rate": factors.get("ignition_rate", ["medium"]),
            "weather_volatility": factors.get("weather_volatility", ["medium"]),
        }

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

    print(f"[*] Results will be saved to: {output_dir}")
    print(f"[*] Generated {len(runs_to_execute)} total run configurations.")
    print(f"[*] Execution mode: {'PARALLEL' if args.parallel else 'SEQUENTIAL'}")
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

    def _process_result(run: dict[str, Any], res: dict[str, Any]) -> None:
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
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(run_single_simulation, *kwargs): run
                for run, kwargs in zip(runs_to_execute, submit_kwargs, strict=True)
            }
            for future in as_completed(futures):
                run = futures[future]
                name = run["name"]
                try:
                    _process_result(run, future.result())
                except Exception as e:
                    print(f"[-] Run '{name}' raised an unhandled exception: {e}")
                    results_key["runs"][name] = {"status": "failed", "error": str(e)}
    else:
        for run, kwargs in zip(runs_to_execute, submit_kwargs, strict=True):
            name = run["name"]
            try:
                _process_result(run, run_single_simulation(*kwargs))
                print(f"[+] Completed: {name}")
            except Exception as e:
                print(f"[-] Run '{name}' failed: {e}")
                results_key["runs"][name] = {"status": "failed", "error": str(e)}

    total_elapsed = time.time() - start_time
    print("=" * 60)
    print(f"[+] All runs finished in {total_elapsed:.1f}s.")

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
        f.write(f"Total Runs: {len(runs_to_execute)}\n")
        f.write(f"Total Elapsed Time: {total_elapsed:.1f}s\n")
        f.write("=" * 60 + "\n\n")
        f.write("\n".join(logs))
    print(f"[+] Consolidated logs written to: {log_file_path}")

    print("\n=== Fire Ecology Baselines Parameter Scan Summary ===")
    print(
        f"{'Run Name':<55} | {'Status':<10} | {'Time (s)':<8} | {'A2 Burned':<10} | {'A2 Cost':<10}"
    )
    print("-" * 105)
    for name, run_res in results_key["runs"].items():
        if run_res.get("status") == "success":
            status = "success"
            elapsed = f"{run_res.get('elapsed_seconds', 0.0):.1f}"
            a2 = run_res["baselines_summary"].get("A2 Centralized", {})
            burned = str(a2.get("burned_cells", "N/A"))
            cost = f"{a2.get('cost', 0.0):,.1f}" if a2 else "N/A"
        else:
            status = "failed"
            elapsed = "N/A"
            burned = "N/A"
            cost = "N/A"
        print(f"{name:<55} | {status:<10} | {elapsed:<8} | {burned:<10} | {cost:<10}")
    print("=" * 105)

    any_failed = any(r.get("status") == "failed" for r in results_key["runs"].values())
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
