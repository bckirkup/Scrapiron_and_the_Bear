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
    include_a4: bool = True
    max_thermal_dim: int | None = None


def _build_fresh_grid(config: ComparisonConfig, rng: np.random.Generator) -> FireGrid:
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


def _evolve_weather(time_step: int, rng: np.random.Generator) -> WeatherState:
    """Deterministic weather evolution (same as adapter)."""
    phase = 2.0 * np.pi * time_step / 200.0
    return WeatherState(
        temperature=25.0 + 10.0 * np.sin(phase) + float(rng.normal(0, 2)),
        humidity=float(np.clip(0.4 + 0.2 * np.cos(phase) + rng.normal(0, 0.05), 0, 1)),
        wind_speed=max(0.0, 5.0 + 3.0 * np.sin(phase * 0.7) + float(rng.normal(0, 1))),
        wind_direction=float((180.0 + 90.0 * np.sin(phase * 0.3)) % 360),
        precipitation=max(0.0, float(rng.exponential(0.5) if rng.random() < 0.1 else 0.0)),
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
        grid = _build_fresh_grid(config, rng)
        opir = OPIRSatellite(cadence=config.opir_cadence)
        metrics = FireMetrics()

        total_det = 0
        total_sup = 0
        total_esc = 0
        total_cost = 0.0

        for step in range(config.steps):
            weather = _evolve_weather(step, weather_rng)
            grid.step(weather, step, rng)
            grid.stochastic_ignition(weather, step, rng)

            result = arch.step(grid, weather, opir, step, rng)
            total_det += len(result.detections)
            total_sup += len(result.suppressions)
            total_esc += result.escalations
            total_cost += result.cost

            # Wire detection latency tracking
            metrics.record_detections_from_grid(result.detections, grid, step)

            step_metrics = StepMetrics(
                time_step=step,
                active_fires=len(grid.active_fire_cells()),
                burned_area=grid.burned_area(),
                detections=len(result.detections),
                suppressions=len(result.suppressions),
                escalations=result.escalations,
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
            )
        )

    return results


def format_comparison_table(results: list[ComparisonResult]) -> str:
    """Format comparison results as an aligned text table."""
    header = (
        f"{'Architecture':<16} {'Detections':>10} {'Suppressions':>12} "
        f"{'Escalations':>11} {'Cost':>8} {'Burned':>8} {'Latency':>8}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        latency_str = f"{r.mean_detection_latency:.1f}" if r.mean_detection_latency < 1e6 else "inf"
        lines.append(
            f"{r.name:<16} {r.detections:>10,} {r.suppressions:>12,} "
            f"{r.escalations:>11,} {r.cost:>8,.1f} {r.burned_cells:>8,} {latency_str:>8}"
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
            }
        )
    return json.dumps(data, indent=2)
