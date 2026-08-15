"""Tests for A4 BMA (TattleTots) architecture."""

from __future__ import annotations

import numpy as np
import pytest

from fire_ecology.architectures.a4_bma import BMAFireEcology
from fire_ecology.drones.body_plan import BodyPlan
from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.opir import OPIRSatellite


def _setup() -> tuple[FireGrid, WeatherState, OPIRSatellite, np.random.Generator]:
    grid = FireGrid(rows=10, cols=10)
    grid.ignite(5, 5, 0)
    weather = WeatherState(temperature=35.0, humidity=0.2, wind_speed=10.0)
    opir = OPIRSatellite(cadence=1)
    rng = np.random.default_rng(42)
    return grid, weather, opir, rng


class TestBMAFireEcology:
    def test_step_returns_result(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = BMAFireEcology(n_drones=5, grid_rows=10, grid_cols=10, seed=42, initial_population=5)
        result = arch.step(grid, weather, opir, time_step=0, rng=rng)
        assert result.cost >= 0.0
        assert isinstance(result.detections, list)
        assert isinstance(result.suppressions, list)
        assert result.initiation_is_degenerate is True
        assert result.initiation_degeneracy_reasons
        assert result.event_prevalence == pytest.approx(1.0, rel=0.0, abs=1e-12)

    def test_multiple_steps(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = BMAFireEcology(n_drones=5, grid_rows=10, grid_cols=10, seed=42, initial_population=5)
        for step in range(10):
            grid.step(weather, step, rng)
            result = arch.step(grid, weather, opir, time_step=step, rng=rng)
            assert result.cost >= 0.0

    def test_living_population(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = BMAFireEcology(n_drones=5, grid_rows=10, grid_cols=10, seed=42, initial_population=5)
        assert arch.living_population == 0  # not initialized yet
        arch.step(grid, weather, opir, time_step=0, rng=rng)
        assert arch.living_population > 0

    def test_reset(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = BMAFireEcology(n_drones=5, grid_rows=10, grid_cols=10, seed=42, initial_population=5)
        arch.step(grid, weather, opir, time_step=0, rng=rng)
        assert arch.world is not None
        arch.reset()
        assert arch.world is None
        assert arch.living_population == 0

    def test_body_plan_determines_suppression(self) -> None:
        arch = BMAFireEcology(
            n_drones=5,
            grid_rows=10,
            grid_cols=10,
            seed=42,
            body_plan=BodyPlan.strike_large(),
            initial_population=5,
        )
        assert arch.body_plan.suppression_effectiveness > 0.5

    def test_uses_harness_grid_not_internal_adapter_physics(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = BMAFireEcology(n_drones=5, grid_rows=10, grid_cols=10, seed=42, initial_population=5)
        arch.step(grid, weather, opir, time_step=0, rng=rng)
        assert arch._adapter is not None
        assert len(grid.active_fire_cells()) > 0
        assert len(arch._adapter.fire_grid.active_fire_cells()) == 0

    def test_opir_ablation_changes_attribution(self) -> None:
        grid_on, weather_on, opir_on, rng_on = _setup()
        enabled = BMAFireEcology(
            n_drones=5,
            grid_rows=10,
            grid_cols=10,
            seed=42,
            initial_population=5,
            use_opir=True,
        )
        enabled.step(grid_on, weather_on, opir_on, time_step=0, rng=rng_on)
        assert enabled.world is not None
        for agent in enabled.world.agents.values():
            agent.state.energy.information = -1e9
            agent.state.energy.attention = -1e9
        enabled_result = enabled.step(grid_on, weather_on, opir_on, time_step=1, rng=rng_on)

        grid_off, weather_off, opir_off, rng_off = _setup()
        disabled = BMAFireEcology(
            n_drones=5,
            grid_rows=10,
            grid_cols=10,
            seed=42,
            initial_population=5,
            use_opir=False,
        )
        disabled.step(grid_off, weather_off, opir_off, time_step=0, rng=rng_off)
        assert disabled.world is not None
        for agent in disabled.world.agents.values():
            agent.state.energy.information = -1e9
            agent.state.energy.attention = -1e9
        disabled_result = disabled.step(grid_off, weather_off, opir_off, time_step=1, rng=rng_off)

        assert enabled_result.opir_detections > 0
        assert disabled_result.opir_detections == 0
        assert enabled_result.detections != disabled_result.detections
        assert enabled_result.ecology_extinct is True
        assert disabled_result.ecology_extinct is True

    def test_opir_ablation_skips_opir_stream_scan(self) -> None:
        grid_on, weather_on, opir_on, rng_on = _setup()
        enabled = BMAFireEcology(
            n_drones=5,
            grid_rows=10,
            grid_cols=10,
            seed=42,
            initial_population=5,
            use_opir=True,
        )
        enabled.step(grid_on, weather_on, opir_on, time_step=0, rng=rng_on)

        grid_off, weather_off, opir_off, rng_off = _setup()
        disabled = BMAFireEcology(
            n_drones=5,
            grid_rows=10,
            grid_cols=10,
            seed=42,
            initial_population=5,
            use_opir=False,
        )
        disabled.step(grid_off, weather_off, opir_off, time_step=0, rng=rng_off)

        assert rng_on.random() != rng_off.random()
