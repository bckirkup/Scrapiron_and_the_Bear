"""Fire ecology simulation runner — shared single/batch entry points for all layers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

from domain_runner.batch import run_batch as execute_batch
from domain_runner.config import deep_merge, load_json
from domain_runner.layer import DomainOnlyLayer, SimulationLayer
from domain_runner.single import print_result_summary, run_simulation_timed
from domain_runner.types import RunContext, SimulationResult

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.metrics.fire_metrics import FireMetrics, StepMetrics

_ADAPTER_KEYS = (
    "grid_rows",
    "grid_cols",
    "seed",
    "n_cameras",
    "n_weather_stations",
    "n_fuel_sensors",
    "opir_cadence",
    "max_thermal_dim",
    "base_ignition_rate",
    "weather_volatility",
    "n_drones",
    "suppression_effectiveness",
)

_DEFAULT_DOMAIN: dict[str, Any] = {
    "grid_rows": 20,
    "grid_cols": 20,
    "seed": 42,
    "steps": 200,
}


def adapter_kwargs(domain_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "grid_rows": domain_cfg.get("grid_rows", 20),
        "grid_cols": domain_cfg.get("grid_cols", 20),
        "seed": domain_cfg.get("seed", 42),
    }
    for key in _ADAPTER_KEYS:
        if key in domain_cfg and key not in kwargs:
            kwargs[key] = domain_cfg[key]
    return kwargs


class FireDomainHooks:
    domain_name = "fire_ecology"
    default_config_path = "configs/domain_default.json"

    def __init__(self) -> None:
        self._metrics = FireMetrics()
        self._layer_name = "domain_only"

    def load_run_context(
        self,
        *,
        config_path: str | None = None,
        cli_overrides: dict[str, Any] | None = None,
    ) -> RunContext:
        raw: dict[str, Any] = {"domain": dict(_DEFAULT_DOMAIN), "layer": "domain_only"}
        if config_path:
            raw = deep_merge(raw, load_json(config_path))

        if cli_overrides:
            if "domain" in cli_overrides:
                raw["domain"] = deep_merge(raw.get("domain", {}), cli_overrides["domain"])
            for key, value in cli_overrides.items():
                if key not in ("domain", "simulation", "layer", "output", "verbose"):
                    raw["domain"][key] = value
            for meta_key in ("simulation", "layer", "output", "verbose"):
                if meta_key in cli_overrides:
                    raw[meta_key] = cli_overrides[meta_key]

        domain_cfg = dict(raw.get("domain", {}))
        steps = int(domain_cfg.pop("steps", _DEFAULT_DOMAIN["steps"]))
        seed = int(domain_cfg.get("seed", 42))
        output = raw.get("output")
        return RunContext(
            steps=steps,
            seed=seed,
            domain_config=domain_cfg,
            layer=str(raw.get("layer", "domain_only")),
            simulation_config=dict(raw.get("simulation", {})),
            verbose=bool(raw.get("verbose", False)),
            output_path=Path(output) if output else None,
        )

    def build_adapter(self, domain_config: dict[str, Any]) -> FireEcologyAdapter:
        return FireEcologyAdapter(**adapter_kwargs(domain_config))

    def print_header(self, adapter: FireEcologyAdapter, run: RunContext) -> None:
        self._layer_name = run.layer
        cfg = run.domain_config
        print(f"=== FireEcology ({run.layer}) ===")
        print(
            f"  Steps: {run.steps}, Grid: {cfg.get('grid_rows')}x{cfg.get('grid_cols')}, "
            f"Drones: {adapter.n_drones}, Seed: {run.seed}"
        )
        print()

    def on_step(self, adapter: FireEcologyAdapter, step: int, layer_events: dict[str, Any]) -> None:
        active = adapter.fire_grid.active_fire_cells()
        burned = adapter.fire_grid.burned_area()
        self._metrics.record_detections_from_grid(
            [(r, c) for r, c in active], adapter.fire_grid, step
        )

        if "cost_dict" in layer_events:
            costs = layer_events["cost_dict"]
        else:
            costs = adapter.compute_costs(
                n_escalations=0,
                n_correct=0,
                n_false_alarms=0,
                n_missed=len(active),
            )

        self._metrics.record_step(
            StepMetrics(
                time_step=step,
                active_fires=len(active),
                burned_area=burned,
                surveillance_cost=costs["surveillance_cost"],
                response_cost=costs["response_cost"],
                damage_cost=costs["damage_cost"],
            )
        )

        outcome_counts = layer_events.get("outcome_counts")
        if outcome_counts:
            self._metrics.record_response_outcomes(
                dispatched=outcome_counts["responses_dispatched"],
                judged_necessary=outcome_counts["responses_judged_necessary"],
                judged_unnecessary=outcome_counts["responses_judged_unnecessary"],
                suppressions=outcome_counts.get("responses_judged_necessary", 0),
            )

    def should_stop(
        self, adapter: FireEcologyAdapter, step: int, layer_events: dict[str, Any]
    ) -> bool:
        return bool(layer_events.get("stop"))

    def print_step(
        self,
        adapter: FireEcologyAdapter,
        step: int,
        layer_events: dict[str, Any],
        *,
        verbose: bool,
    ) -> None:
        if not verbose or step % 50 != 0:
            return
        active = adapter.fire_grid.active_fire_cells()
        if layer_events.get("population") is not None:
            print(
                f"  Step {step:4d}: pop={layer_events['population']:3d} "
                f"fires={len(active)} reports={layer_events.get('reports_issued', 0)}"
            )
        else:
            print(f"  Step {step:4d}: fires={len(active)} burned={adapter.fire_grid.burned_area()}")

    def summarize(
        self, adapter: FireEcologyAdapter, layer_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        summary = dict(self._metrics.summary())
        if "telemetry_summary" in layer_metrics:
            summary["ecology"] = layer_metrics["telemetry_summary"]
        if "cost_summary" in layer_metrics:
            summary["costs"] = layer_metrics["cost_summary"]
        return summary

    def write_output(self, result: SimulationResult, path: str) -> None:
        if "simulation_output" in result.layer_metrics:
            output = result.layer_metrics["simulation_output"]
            output.run_summary.wall_time_seconds = result.wall_time_seconds
            output.domain_metrics = result.domain_metrics
            output.write_json(path)
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)


def resolve_layer(name: str) -> SimulationLayer:
    if name in ("domain_only", "domain", "none"):
        return DomainOnlyLayer()
    if name in ("tattletots", "tots"):
        from tattletots.integration.tattletots_layer import TattleTotsLayer

        return TattleTotsLayer()
    raise ValueError(f"Unknown layer {name!r}. Supported: domain_only, tattletots")


def run_fire_simulation(run: RunContext) -> SimulationResult:
    hooks = FireDomainHooks()
    layer = resolve_layer(run.layer)
    result = run_simulation_timed(hooks, layer, run)
    print_result_summary(result)
    return result


def run_fire_batch_entry(
    name: str,
    run_config: dict[str, Any],
    output_dir: Path,
    verbose: bool,
) -> dict[str, Any]:
    layer_name = str(run_config.pop("_layer", "domain_only"))
    simulation_config = dict(run_config.pop("simulation", {}))
    steps = int(run_config.pop("steps", _DEFAULT_DOMAIN["steps"]))
    seed = int(run_config.get("seed", 42))

    run = RunContext(
        steps=steps,
        seed=seed,
        domain_config=run_config,
        layer=layer_name,
        simulation_config=simulation_config,
        verbose=verbose,
        output_path=output_dir / f"{name}_results.json",
    )

    resolved_cfg = output_dir / f"{name}_config.json"
    with open(resolved_cfg, "w", encoding="utf-8") as f:
        json.dump(
            {
                "domain": {**run_config, "steps": steps},
                "layer": layer_name,
                "simulation": simulation_config,
            },
            f,
            indent=2,
        )

    start = time.time()
    try:
        result = run_fire_simulation(run)
        return {
            "status": "success",
            "layer": layer_name,
            "elapsed_seconds": time.time() - start,
            "metrics": result.domain_metrics,
            "output_file": run.output_path.name if run.output_path else None,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "layer": layer_name,
            "elapsed_seconds": time.time() - start,
            "error": str(exc),
        }


def run_fire_batch(
    batch_config_path: Path,
    *,
    output_dir: Path | None = None,
    parallel: bool = False,
    workers: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    batch = load_json(batch_config_path)
    out = Path(output_dir or batch.get("output_directory", "batch_results"))
    default: dict[str, Any] = {"domain": dict(_DEFAULT_DOMAIN)}
    if "simulation" in batch:
        default["simulation"] = batch["simulation"]
    return cast(
        dict[str, Any],
        execute_batch(
            batch,
            run_fire_batch_entry,
            output_dir=out,
            default_config=default,
            parallel=parallel,
            workers=workers,
            verbose=verbose,
        ),
    )
