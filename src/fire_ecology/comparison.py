"""Head-to-head comparison harness for all architectures.

Runs A0–A4 on the *same* fire scenario (same seed, same grid) and
produces a summary table for falsification analysis per spec §10.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from fire_ecology.architectures.a0_human import HumanBaseline
from fire_ecology.architectures.a1_camera_ml import CameraMLNetwork
from fire_ecology.architectures.a2_centralized import CentralizedOptimizer
from fire_ecology.architectures.a3_federated import FederatedEdge
from fire_ecology.architectures.a4_bma import BMAFireEcology
from fire_ecology.architectures.base import Architecture
from fire_ecology.architectures.ecology_options import EcologyMeasurementOptions
from fire_ecology.drones.body_plan import BodyPlan
from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.metrics.fire_metrics import FireMetrics, StepMetrics
from fire_ecology.sensors.camera_tower import CameraTower
from fire_ecology.sensors.opir import OPIRSatellite


@dataclass
class ComparisonResult:
    """Summary for a single architecture's run."""

    name: str
    detections: int = 0
    suppressions: int = 0
    escalations: int = 0
    cost: float = 0.0
    burned_cells: int = 0
    mean_detection_latency: float = float("inf")
    suppression_success_rate: float = 0.0
    tot_detections: int = 0
    opir_detections: int = 0
    opir_shadow_detections: int = 0
    opir_backstop_ablated: bool = False
    reports_issued: int = 0
    correct_reports: int = 0
    false_alarms: int = 0
    living_population_trajectory: list[int] = field(default_factory=list)
    ecology_extinct: bool = False
    event_prevalence: float = 0.0
    grounded_yield_share: float = 0.0
    attention_solvent_fraction: float = 0.0
    mean_attention_carrying_capacity: float = 0.0
    initiation_is_degenerate: bool = False
    initiation_degeneracy_reasons: list[str] = field(default_factory=list)


@dataclass
class ComparisonConfig:
    """Configuration for a head-to-head comparison run."""

    steps: int = 100
    grid_rows: int = 20
    grid_cols: int = 20
    seed: int = 42
    n_drones: int = 10
    n_cameras: int = 3
    opir_cadence: int = 5
    base_ignition_rate: float = 0.0001
    weather_volatility: float = 1.0
    include_a4: bool = True
    include_a4_opir_ablation: bool = False
    max_thermal_dim: int | None = None
    a4_options: EcologyMeasurementOptions = field(default_factory=EcologyMeasurementOptions)


def build_fresh_grid(config: ComparisonConfig, _rng: np.random.Generator) -> FireGrid:
    """Create and seed a fire grid with a deterministic initial ignition."""
    grid = FireGrid(rows=config.grid_rows, cols=config.grid_cols)
    mid_r, mid_c = config.grid_rows // 2, config.grid_cols // 2
    grid.ignite(mid_r, mid_c, time_step=0)
    return grid


def _build_cameras(config: ComparisonConfig) -> list[CameraTower]:
    cameras: list[CameraTower] = []
    for i in range(config.n_cameras):
        r = int((i + 1) * config.grid_rows / (config.n_cameras + 1))
        c = int((i + 1) * config.grid_cols / (config.n_cameras + 1))
        cameras.append(CameraTower(row=r, col=c, max_range=10.0))
    return cameras


def evolve_weather(
    time_step: int, rng: np.random.Generator, volatility: float = 1.0
) -> WeatherState:
    """Deterministic weather evolution (same as adapter)."""
    phase = 2.0 * np.pi * time_step / 200.0
    v = volatility
    return WeatherState(
        temperature=25.0 + 10.0 * np.sin(phase) + float(rng.normal(0, 2 * v)),
        humidity=float(np.clip(0.4 + 0.2 * np.cos(phase) + rng.normal(0, 0.05 * v), 0, 1)),
        wind_speed=max(0.0, 5.0 + 3.0 * np.sin(phase * 0.7) + float(rng.normal(0, 1 * v))),
        wind_direction=float((180.0 + 90.0 * np.sin(phase * 0.3)) % 360),
        precipitation=max(0.0, float(rng.exponential(0.5 * v) if rng.random() < 0.1 else 0.0)),
    )


def _make_architectures(
    config: ComparisonConfig,
    cameras: list[CameraTower],
) -> list[tuple[str, Architecture]]:
    """Instantiate all architectures with identical hardware."""
    archs: list[tuple[str, Architecture]] = [
        ("A0 Human", HumanBaseline()),
        ("A1 Camera ML", CameraMLNetwork(cameras=cameras)),
        (
            "A2 Centralized",
            CentralizedOptimizer(
                n_drones=config.n_drones,
                body_plan=BodyPlan.strike_small(),
            ),
        ),
        (
            "A3 Federated",
            FederatedEdge(
                n_nodes=config.n_drones,
                body_plan=BodyPlan.strike_small(),
            ),
        ),
    ]
    if config.include_a4:
        archs.append(
            (
                "A4 BMA",
                BMAFireEcology(
                    n_drones=config.n_drones,
                    grid_rows=config.grid_rows,
                    grid_cols=config.grid_cols,
                    seed=config.seed,
                    body_plan=BodyPlan.hybrid(),
                    initial_population=config.n_drones,
                    max_thermal_dim=config.max_thermal_dim,
                    use_opir=True,
                    n_cameras=config.n_cameras,
                    opir_cadence=config.opir_cadence,
                    options=config.a4_options,
                ),
            )
        )
        if config.include_a4_opir_ablation:
            archs.append(
                (
                    "A4 BMA (OPIR ablated)",
                    BMAFireEcology(
                        n_drones=config.n_drones,
                        grid_rows=config.grid_rows,
                        grid_cols=config.grid_cols,
                        seed=config.seed,
                        body_plan=BodyPlan.hybrid(),
                        initial_population=config.n_drones,
                        max_thermal_dim=config.max_thermal_dim,
                        use_opir=False,
                        n_cameras=config.n_cameras,
                        opir_cadence=config.opir_cadence,
                    ),
                )
            )
    return archs


@dataclass
class _RunState:
    """Mutable state for a single architecture run."""

    grid: FireGrid
    metrics: FireMetrics = field(default_factory=FireMetrics)
    total_detections: int = 0
    total_suppressions: int = 0
    total_escalations: int = 0
    total_cost: float = 0.0


def run_comparison(config: ComparisonConfig | None = None) -> list[ComparisonResult]:
    """Execute head-to-head comparison, returning per-architecture summaries."""
    if config is None:
        config = ComparisonConfig()

    cameras = _build_cameras(config)
    archs = _make_architectures(config, cameras)
    results: list[ComparisonResult] = []

    for name, arch in archs:
        rng = np.random.default_rng(config.seed)
        weather_rng = np.random.default_rng(config.seed + 1)
        grid = build_fresh_grid(config, rng)
        opir = OPIRSatellite(cadence=config.opir_cadence)
        metrics = FireMetrics()

        total_det = 0
        total_sup = 0
        total_esc = 0
        total_cost = 0.0
        living_population_trajectory: list[int] = []
        total_tot_detections = 0
        total_opir_detections = 0
        total_opir_shadow_detections = 0
        total_reports_issued = 0
        total_correct_reports = 0
        total_false_alarms = 0
        opir_backstop_ablated = False
        initiation_is_degenerate = False
        initiation_degeneracy_reasons: list[str] = []
        event_prevalence = 0.0
        grounded_yield_share = 0.0
        attention_solvent_fraction = 0.0
        mean_attention_carrying_capacity = 0.0

        for step in range(config.steps):
            weather = evolve_weather(step, weather_rng, config.weather_volatility)
            grid.step(weather, step, rng)
            grid.stochastic_ignition(weather, step, rng, base_rate=config.base_ignition_rate)

            result = arch.step(grid, weather, opir, step, rng)
            total_det += len(result.detections)
            total_sup += len(result.suppressions)
            total_esc += result.escalations
            total_cost += result.cost
            total_tot_detections += result.tot_detections
            total_opir_detections += result.opir_detections
            total_opir_shadow_detections += result.opir_shadow_detections
            total_reports_issued += result.reports_issued
            total_correct_reports += result.correct_reports
            total_false_alarms += result.false_alarms
            opir_backstop_ablated = result.opir_backstop_ablated
            event_prevalence = result.event_prevalence
            grounded_yield_share = result.grounded_yield_share
            attention_solvent_fraction = result.attention_solvent_fraction
            mean_attention_carrying_capacity = result.mean_attention_carrying_capacity
            initiation_is_degenerate = result.initiation_is_degenerate
            initiation_degeneracy_reasons = list(result.initiation_degeneracy_reasons)
            if result.living_population is not None:
                living_population_trajectory.append(result.living_population)

            # Wire detection latency tracking
            metrics.record_detections_from_grid(result.detections, grid, step)

            step_metrics = StepMetrics(
                time_step=step,
                active_fires=len(grid.active_fire_cells()),
                burned_area=grid.burned_area(),
                detections=len(result.detections),
                suppressions=len(result.suppressions),
                escalations=result.escalations,
                tot_detections=result.tot_detections,
                opir_detections=result.opir_detections,
                living_population=result.living_population,
                surveillance_cost=result.cost * 0.3,
                response_cost=result.cost * 0.5,
                damage_cost=result.cost * 0.2,
            )
            metrics.record_step(step_metrics)

        arch.reset()

        results.append(
            ComparisonResult(
                name=name,
                detections=total_det,
                suppressions=total_sup,
                escalations=total_esc,
                cost=round(total_cost, 1),
                burned_cells=grid.burned_area(),
                mean_detection_latency=round(metrics.mean_detection_latency, 2),
                suppression_success_rate=round(metrics.suppression_success_rate, 4),
                tot_detections=total_tot_detections,
                opir_detections=total_opir_detections,
                opir_shadow_detections=total_opir_shadow_detections,
                opir_backstop_ablated=opir_backstop_ablated,
                reports_issued=total_reports_issued,
                correct_reports=total_correct_reports,
                false_alarms=total_false_alarms,
                living_population_trajectory=living_population_trajectory,
                ecology_extinct=any(population == 0 for population in living_population_trajectory),
                event_prevalence=event_prevalence,
                grounded_yield_share=grounded_yield_share,
                attention_solvent_fraction=attention_solvent_fraction,
                mean_attention_carrying_capacity=mean_attention_carrying_capacity,
                initiation_is_degenerate=initiation_is_degenerate,
                initiation_degeneracy_reasons=initiation_degeneracy_reasons,
            )
        )

    return results


def format_comparison_table(results: list[ComparisonResult]) -> str:
    """Format comparison results as an aligned text table."""
    header = (
        f"{'Architecture':<16} {'Detections':>10} {'Suppressions':>12} "
        f"{'Tot':>7} {'OPIR':>7} {'Escalations':>11} {'Cost':>8} "
        f"{'Burned':>8} {'Latency':>8} {'Extinct':>8}"
        f" {'Degenerate':>10}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        latency_str = f"{r.mean_detection_latency:.1f}" if r.mean_detection_latency < 1e6 else "inf"
        lines.append(
            f"{r.name:<16} {r.detections:>10,} {r.suppressions:>12,} "
            f"{r.tot_detections:>7,} {r.opir_detections:>7,} "
            f"{r.escalations:>11,} {r.cost:>8,.1f} {r.burned_cells:>8,} "
            f"{latency_str:>8} {str(r.ecology_extinct):>8} {str(r.initiation_is_degenerate):>10}"
        )
    return "\n".join(lines)


def format_comparison_json(results: list[ComparisonResult]) -> str:
    """Format comparison results as JSON."""
    data = []
    for r in results:
        data.append(
            {
                "architecture": r.name,
                "detections": r.detections,
                "suppressions": r.suppressions,
                "escalations": r.escalations,
                "cost": r.cost,
                "burned_cells": r.burned_cells,
                "mean_detection_latency": r.mean_detection_latency,
                "suppression_success_rate": r.suppression_success_rate,
                "tot_detections": r.tot_detections,
                "opir_detections": r.opir_detections,
                "opir_shadow_detections": r.opir_shadow_detections,
                "opir_backstop_ablated": r.opir_backstop_ablated,
                "reports_issued": r.reports_issued,
                "correct_reports": r.correct_reports,
                "false_alarms": r.false_alarms,
                "tot_detection_share": (
                    r.tot_detections / (r.tot_detections + r.opir_detections)
                    if r.tot_detections + r.opir_detections
                    else 0.0
                ),
                "living_population_trajectory": r.living_population_trajectory,
                "ecology_extinct": r.ecology_extinct,
                "event_prevalence": r.event_prevalence,
                "grounded_yield_share": r.grounded_yield_share,
                "attention_solvent_fraction": r.attention_solvent_fraction,
                "mean_attention_carrying_capacity": r.mean_attention_carrying_capacity,
                "initiation_is_degenerate": r.initiation_is_degenerate,
                "initiation_degeneracy_reasons": r.initiation_degeneracy_reasons,
            }
        )
    return json.dumps(data, indent=2)
