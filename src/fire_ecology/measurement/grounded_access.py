"""Measure agent-only fire detection against grounded raw-stream access.

The A4 arm used to be unfalsifiable through the domain detection metrics: the
OPIR backstop was appended to every step's detections, so detection rate,
latency and suppression counts survived total Tot extinction.  This harness
runs the A4 ecology with the backstop ablated (see
:class:`~fire_ecology.architectures.ecology_options.EcologyMeasurementOptions`)
so the reported detections are produced by agent reports alone, and sweeps the
TattleTots ``grounded_input_fraction`` knob at a fixed seed.

Every rate is emitted next to its null: report precision is reported with the
engine's chance precision and static-prior precision, and with the adapter's
instrument-level ``chance_baseline`` / ``static_prior_baseline`` from
``tattletots.interface.instrument.validate_instrument``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tattletots.engine.world import World
from tattletots.interface.instrument import validate_instrument
from tattletots.output_schema import (
    CostMetrics,
    EcologyMetrics,
    RunSummary,
    SimulationOutput,
    TimeSeries,
)

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.architectures.a4_bma import BMAFireEcology
from fire_ecology.architectures.base import ArchitectureResult
from fire_ecology.architectures.ecology_options import EcologyMeasurementOptions
from fire_ecology.comparison import ComparisonConfig, build_fresh_grid, evolve_weather
from fire_ecology.drones.body_plan import BodyPlan
from fire_ecology.environment.fire import FireGrid
from fire_ecology.sensors.opir import OPIRSatellite

DEFAULT_GROUNDED_FRACTIONS: tuple[float, ...] = (0.0, 0.34, 0.67)


@dataclass(frozen=True)
class ArmSpec:
    """One measurement arm: a grounded-access setting and a detection path."""

    label: str
    grounded_input_fraction: float
    ablate_opir_backstop: bool = True
    grounded_attractiveness_multiplier: float = 1.0

    def options(self) -> EcologyMeasurementOptions:
        return EcologyMeasurementOptions(
            ablate_opir_backstop=self.ablate_opir_backstop,
            grounded_input_fraction=self.grounded_input_fraction,
            grounded_attractiveness_multiplier=self.grounded_attractiveness_multiplier,
        )

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, **self.options().as_dict()}


@dataclass(frozen=True)
class SweepSpec:
    """Shared scenario configuration; identical across arms."""

    steps: int = 200
    grid_rows: int = 20
    grid_cols: int = 20
    seed: int = 42
    n_drones: int = 10
    n_cameras: int = 3
    opir_cadence: int = 5
    instrument_steps: int = 200

    def comparison_config(self, options: EcologyMeasurementOptions) -> ComparisonConfig:
        return ComparisonConfig(
            steps=self.steps,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            seed=self.seed,
            n_drones=self.n_drones,
            n_cameras=self.n_cameras,
            opir_cadence=self.opir_cadence,
            a4_options=options,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "seed": self.seed,
            "n_drones": self.n_drones,
            "n_cameras": self.n_cameras,
            "opir_cadence": self.opir_cadence,
        }


@dataclass
class _Tally:
    """Running detection and report counts for one arm."""

    active_fire_cell_steps: int = 0
    steps_with_fire: int = 0
    steps_with_agent_detection: int = 0
    agent_detection_cell_steps: int = 0
    opir_detection_cell_steps: int = 0
    opir_shadow_cell_steps: int = 0
    reports_issued: int = 0
    correct_reports: int = 0
    false_alarms: int = 0
    suppressions: int = 0
    cost_per_step: list[float] = field(default_factory=list)
    agent_detected_cells: set[tuple[int, int]] = field(default_factory=set)
    burning_cells: set[tuple[int, int]] = field(default_factory=set)

    def record(self, result: ArchitectureResult, active: Sequence[tuple[int, int]]) -> None:
        self.active_fire_cell_steps += len(active)
        self.burning_cells.update(active)
        if active:
            self.steps_with_fire += 1
        agent_cells = result.detections[: result.tot_detections]
        self.agent_detected_cells.update(agent_cells)
        self.agent_detection_cell_steps += result.tot_detections
        self.opir_detection_cell_steps += result.opir_detections
        self.opir_shadow_cell_steps += result.opir_shadow_detections
        if result.tot_detections > 0:
            self.steps_with_agent_detection += 1
        self.reports_issued += result.reports_issued
        self.correct_reports += result.correct_reports
        self.false_alarms += result.false_alarms
        self.suppressions += len(result.suppressions)
        self.cost_per_step.append(result.cost)


def _ratio(numerator: float, denominator: float) -> float:
    """Safe ratio; zero when the denominator is empty."""
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def reproductive_correlation(parents_by_agent: Mapping[str, Sequence[str]]) -> float | None:
    """Pearson correlation between a parent's and its child's offspring counts.

    ``parents_by_agent`` maps every agent that ever existed to its parent ids.
    Founders have no parents and contribute no pair.  Returns ``None`` when
    fewer than three parent-child pairs exist or either side has zero variance,
    because a correlation is not defined there.
    """
    offspring: Counter[str] = Counter()
    for parents in parents_by_agent.values():
        for parent_id in parents:
            offspring[parent_id] += 1

    pairs = [
        (offspring[parent_id], offspring[child_id])
        for child_id, parents in parents_by_agent.items()
        for parent_id in parents
        if parent_id in parents_by_agent
    ]
    if len(pairs) < 3:
        return None
    parent_counts = np.array([p for p, _ in pairs], dtype=np.float64)
    child_counts = np.array([c for _, c in pairs], dtype=np.float64)
    if math.isclose(float(parent_counts.std()), 0.0) or math.isclose(
        float(child_counts.std()), 0.0
    ):
        return None
    return float(np.corrcoef(parent_counts, child_counts)[0, 1])


def _degeneracy_reasons(telemetry: Mapping[str, Any]) -> list[str]:
    """Initiation-degeneracy reasons reported by the engine, as plain strings."""
    raw: object = telemetry.get("initiation_degeneracy_reasons", [])
    if not isinstance(raw, list):
        return []
    return [str(reason) for reason in raw]


def _world_parents(world: World | None) -> dict[str, list[str]]:
    if world is None:
        return {}
    return {agent.id: list(agent.state.parent_ids) for agent in world.agents.values()}


def _run_steps(
    arch: BMAFireEcology,
    grid: FireGrid,
    config: ComparisonConfig,
) -> _Tally:
    """Advance the fire physics and the A4 ecology for the configured steps."""
    rng = np.random.default_rng(config.seed)
    weather_rng = np.random.default_rng(config.seed + 1)
    opir = OPIRSatellite(cadence=config.opir_cadence)
    tally = _Tally()

    for step in range(config.steps):
        weather = evolve_weather(step, weather_rng, config.weather_volatility)
        grid.step(weather, step, rng)
        grid.stochastic_ignition(weather, step, rng, base_rate=config.base_ignition_rate)
        active = grid.active_fire_cells()
        result = arch.step(grid, weather, opir, step, rng)
        tally.record(result, active)
    return tally


def _detection_metrics(tally: _Tally, spec: ArmSpec) -> dict[str, Any]:
    """Agent-only and OPIR-assisted detection metrics for one arm."""
    assisted_cell_steps = tally.agent_detection_cell_steps + tally.opir_detection_cell_steps
    counterfactual_cell_steps = (
        tally.agent_detection_cell_steps + tally.opir_detection_cell_steps
    ) + tally.opir_shadow_cell_steps
    return {
        "opir_backstop_ablated": spec.ablate_opir_backstop,
        "active_fire_cell_steps": tally.active_fire_cell_steps,
        "steps_with_fire": tally.steps_with_fire,
        "agent_only_detection_cell_steps": tally.agent_detection_cell_steps,
        "agent_only_detection_rate": _ratio(
            tally.agent_detection_cell_steps, tally.active_fire_cell_steps
        ),
        "agent_only_step_detection_rate": _ratio(
            tally.steps_with_agent_detection, tally.steps_with_fire
        ),
        "agent_only_cell_recall": _ratio(len(tally.agent_detected_cells), len(tally.burning_cells)),
        "opir_detection_cell_steps": tally.opir_detection_cell_steps,
        "opir_shadow_detection_cell_steps": tally.opir_shadow_cell_steps,
        "reported_detection_rate": _ratio(assisted_cell_steps, tally.active_fire_cell_steps),
        "opir_assisted_detection_rate_counterfactual": _ratio(
            counterfactual_cell_steps, tally.active_fire_cell_steps
        ),
        "suppressions": tally.suppressions,
    }


def _report_metrics(tally: _Tally, telemetry: Mapping[str, Any]) -> dict[str, Any]:
    """Report-level precision and false-alarm rate, each next to its null."""
    return {
        "reports_issued": tally.reports_issued,
        "correct_reports": tally.correct_reports,
        "false_alarms": tally.false_alarms,
        "report_precision": _ratio(tally.correct_reports, tally.reports_issued),
        "report_false_alarm_rate": _ratio(tally.false_alarms, tally.reports_issued),
        "null_chance_precision": float(telemetry["chance_precision"]),
        "null_static_prior_precision": float(telemetry["static_prior_precision"]),
    }


def _solvency_metrics(telemetry: Mapping[str, Any], series: TimeSeries) -> dict[str, Any]:
    """Per-capita attention solvency and grounded-yield share."""
    per_capita_capacity = [
        capacity / population
        for capacity, population in zip(
            series.attention_carrying_capacity, series.population, strict=False
        )
        if population > 0
    ]
    solvent_share = [
        solvent / eligible
        for solvent, eligible in zip(
            series.n_attention_solvent_agents, series.n_attention_eligible_agents, strict=False
        )
        if eligible > 0
    ]
    return {
        "attention_solvent_fraction": float(telemetry["attention_solvent_fraction"]),
        "mean_attention_carrying_capacity": float(telemetry["mean_attention_carrying_capacity"]),
        "mean_per_capita_attention_capacity": (
            float(np.mean(per_capita_capacity)) if per_capita_capacity else 0.0
        ),
        "mean_solvent_share_of_eligible": (float(np.mean(solvent_share)) if solvent_share else 0.0),
        "grounded_yield_share": float(telemetry["grounded_yield_share"]),
        "effective_grounded_yield_share": float(telemetry["effective_grounded_yield_share"]),
    }


def _ecology_metrics(telemetry: Mapping[str, Any]) -> EcologyMetrics:
    fields = set(EcologyMetrics.model_fields)
    return EcologyMetrics(**{k: v for k, v in telemetry.items() if k in fields})


def _cost_metrics(tally: _Tally) -> CostMetrics:
    total = float(sum(tally.cost_per_step))
    steps = max(len(tally.cost_per_step), 1)
    return CostMetrics(
        total_surveillance_cost=total * 0.3,
        total_response_cost=total * 0.5,
        total_damage_cost=total * 0.2,
        total_cost=total,
        mean_cost_per_step=total / steps,
    )


def instrument_nulls(sweep: SweepSpec) -> dict[str, Any]:
    """Instrument-level nulls for the fire adapter, independent of the arms."""
    adapter = FireEcologyAdapter(
        grid_rows=sweep.grid_rows,
        grid_cols=sweep.grid_cols,
        seed=sweep.seed,
        n_cameras=sweep.n_cameras,
        opir_cadence=sweep.opir_cadence,
    )
    report = validate_instrument(adapter, steps=sweep.instrument_steps)
    return {
        "measured_steps": report.measured_steps,
        "event_steps": report.event_steps,
        "distinct_event_locations": report.distinct_event_locations,
        "n_candidate_locations": len(report.candidate_locations),
        "inferability_precision": report.inferability_precision,
        "decoder_precision": report.decoder_precision,
        "null_static_prior_baseline": report.static_prior_baseline,
        "null_chance_baseline": report.chance_baseline,
        "findings": [str(finding) for finding in report.findings],
    }


@dataclass
class ArmResult:
    """One arm's unified output plus the flat numbers used for comparison."""

    spec: ArmSpec
    output: SimulationOutput

    def key_numbers(self) -> dict[str, Any]:
        metrics = self.output.domain_metrics
        detection = metrics["detection"]
        reports = metrics["reports"]
        solvency = metrics["solvency"]
        return {
            "label": self.spec.label,
            "grounded_input_fraction": self.spec.grounded_input_fraction,
            "opir_backstop_ablated": self.spec.ablate_opir_backstop,
            "agent_only_detection_rate": detection["agent_only_detection_rate"],
            "agent_only_step_detection_rate": detection["agent_only_step_detection_rate"],
            "agent_only_cell_recall": detection["agent_only_cell_recall"],
            "reported_detection_rate": detection["reported_detection_rate"],
            "report_precision": reports["report_precision"],
            "report_false_alarm_rate": reports["report_false_alarm_rate"],
            "null_chance_precision": reports["null_chance_precision"],
            "null_static_prior_precision": reports["null_static_prior_precision"],
            "attention_solvent_fraction": solvency["attention_solvent_fraction"],
            "mean_per_capita_attention_capacity": solvency["mean_per_capita_attention_capacity"],
            "grounded_yield_share": solvency["grounded_yield_share"],
            "effective_grounded_yield_share": solvency["effective_grounded_yield_share"],
            "parent_child_reproductive_correlation": metrics[
                "parent_child_reproductive_correlation"
            ],
            "final_population": self.output.ecology_metrics.final_population,
            "ecology_extinct": metrics["ecology_extinct"],
        }


def run_arm(spec: ArmSpec, sweep: SweepSpec, *, nulls: dict[str, Any] | None = None) -> ArmResult:
    """Run a single measurement arm and build its ``SimulationOutput``."""
    config = sweep.comparison_config(spec.options())
    arch = BMAFireEcology(
        n_drones=sweep.n_drones,
        grid_rows=sweep.grid_rows,
        grid_cols=sweep.grid_cols,
        seed=sweep.seed,
        body_plan=BodyPlan.hybrid(),
        initial_population=sweep.n_drones,
        use_opir=True,
        n_cameras=sweep.n_cameras,
        opir_cadence=sweep.opir_cadence,
        options=spec.options(),
    )
    grid = build_fresh_grid(config, np.random.default_rng(sweep.seed))
    tally = _run_steps(arch, grid, config)

    world = arch.world
    assert world is not None
    telemetry = dict(world.telemetry.summary())
    series = TimeSeries.from_telemetry(world.telemetry, tally.cost_per_step)

    output = SimulationOutput(
        run_summary=RunSummary(domain="fire_ecology", steps_completed=sweep.steps, seed=sweep.seed),
        simulation_config={
            **spec.options().as_dict(),
            "engine_kwargs": spec.options().engine_kwargs(),
        },
        domain_config={
            **sweep.as_dict(),
            "architecture": "A4 BMA",
            "burned_cells": grid.burned_area(),
        },
        ecology_metrics=_ecology_metrics(telemetry),
        cost_metrics=_cost_metrics(tally),
        domain_metrics={
            "arm": spec.as_dict(),
            "detection": _detection_metrics(tally, spec),
            "reports": _report_metrics(tally, telemetry),
            "solvency": _solvency_metrics(telemetry, series),
            "instrument_nulls": nulls if nulls is not None else instrument_nulls(sweep),
            "parent_child_reproductive_correlation": reproductive_correlation(
                _world_parents(world)
            ),
            "ecology_extinct": world.living_population == 0,
            "burned_cells": grid.burned_area(),
            "initiation_is_degenerate": bool(telemetry["initiation_is_degenerate"]),
            "initiation_degeneracy_reasons": _degeneracy_reasons(telemetry),
        },
        time_series=series,
    )
    arch.reset()
    return ArmResult(spec=spec, output=output)


def sweep_arms(
    fractions: Sequence[float] = DEFAULT_GROUNDED_FRACTIONS,
    *,
    include_assisted: bool = True,
) -> list[ArmSpec]:
    """Ablated arms for each grounded fraction, plus their OPIR-assisted twins."""
    arms: list[ArmSpec] = [
        ArmSpec(
            label=f"gif{fraction:.2f}_opir_ablated",
            grounded_input_fraction=fraction,
            ablate_opir_backstop=True,
        )
        for fraction in fractions
    ]
    if include_assisted:
        arms.extend(
            ArmSpec(
                label=f"gif{fraction:.2f}_opir_assisted",
                grounded_input_fraction=fraction,
                ablate_opir_backstop=False,
            )
            for fraction in fractions
        )
    return arms


def run_sweep(
    sweep: SweepSpec | None = None,
    arms: Sequence[ArmSpec] | None = None,
) -> list[ArmResult]:
    """Run every arm at the same seed and scenario, sharing the instrument nulls."""
    sweep = sweep or SweepSpec()
    specs = list(arms) if arms is not None else sweep_arms()
    nulls = instrument_nulls(sweep)
    return [run_arm(spec, sweep, nulls=nulls) for spec in specs]
