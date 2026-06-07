"""Tests for fire dynamics."""

from __future__ import annotations

import numpy as np

from fire_ecology.environment.fire import CellFireState, FireGrid, FireState
from fire_ecology.environment.weather import WeatherState


class TestFireState:
    def test_default_unburned(self) -> None:
        fs = FireState()
        assert fs.state == CellFireState.UNBURNED
        assert not fs.is_active

    def test_burning_is_active(self) -> None:
        fs = FireState(state=CellFireState.BURNING)
        assert fs.is_active

    def test_smoldering_is_active(self) -> None:
        fs = FireState(state=CellFireState.SMOLDERING)
        assert fs.is_active

    def test_burned_out_not_active(self) -> None:
        fs = FireState(state=CellFireState.BURNED_OUT)
        assert not fs.is_active


class TestFireGrid:
    def _make_grid(self, rows: int = 10, cols: int = 10) -> FireGrid:
        return FireGrid(rows=rows, cols=cols)

    def test_init_creates_correct_dimensions(self) -> None:
        grid = self._make_grid(5, 8)
        assert len(grid.terrain) == 5
        assert len(grid.terrain[0]) == 8
        assert len(grid.fuel) == 5
        assert len(grid.fire) == 5

    def test_ignite_valid_cell(self) -> None:
        grid = self._make_grid()
        assert grid.ignite(5, 5, 0)
        assert grid.fire[5][5].state == CellFireState.BURNING

    def test_ignite_water_cell_fails(self) -> None:
        grid = self._make_grid()
        grid.terrain[3][3].is_water = True
        assert not grid.ignite(3, 3, 0)

    def test_ignite_already_burning_fails(self) -> None:
        grid = self._make_grid()
        grid.ignite(5, 5, 0)
        assert not grid.ignite(5, 5, 1)

    def test_ignite_out_of_bounds_fails(self) -> None:
        grid = self._make_grid()
        assert not grid.ignite(-1, 5, 0)
        assert not grid.ignite(5, 100, 0)

    def test_active_fire_cells(self) -> None:
        grid = self._make_grid()
        grid.ignite(2, 3, 0)
        grid.ignite(7, 8, 0)
        active = grid.active_fire_cells()
        assert len(active) == 2
        assert (2, 3) in active
        assert (7, 8) in active

    def test_burned_area(self) -> None:
        grid = self._make_grid()
        assert grid.burned_area() == 0
        grid.ignite(5, 5, 0)
        assert grid.burned_area() == 1

    def test_suppress_extinguishes(self) -> None:
        grid = self._make_grid()
        grid.ignite(5, 5, 0)
        result = grid.suppress(5, 5, effectiveness=0.99)
        assert result
        assert grid.fire[5][5].state == CellFireState.BURNED_OUT

    def test_suppress_unburned_fails(self) -> None:
        grid = self._make_grid()
        assert not grid.suppress(5, 5)

    def test_step_spreads_fire(self) -> None:
        rng = np.random.default_rng(42)
        grid = self._make_grid()
        grid.ignite(5, 5, 0)
        weather = WeatherState(wind_speed=15.0, humidity=0.1, temperature=40.0)
        total_spread = 0
        for step in range(20):
            total_spread += grid.step(weather, step + 1, rng)
        assert total_spread > 0

    def test_stochastic_ignition(self) -> None:
        rng = np.random.default_rng(123)
        grid = self._make_grid(rows=50, cols=50)
        weather = WeatherState(temperature=40.0, humidity=0.1, wind_speed=20.0)
        total_ignitions = 0
        for step in range(100):
            ignited = grid.stochastic_ignition(weather, step, rng)
            total_ignitions += len(ignited)
        assert total_ignitions > 0
