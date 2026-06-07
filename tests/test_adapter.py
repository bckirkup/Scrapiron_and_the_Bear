"""Tests for the FireEcology domain adapter."""

from __future__ import annotations

import numpy as np

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter


class TestFireEcologyAdapter:
    def test_get_streams(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10)
        streams = adapter.get_streams()
        assert len(streams) == 3
        assert all(s.dimensionality > 0 for s in streams)

    def test_get_users(self) -> None:
        adapter = FireEcologyAdapter()
        users = adapter.get_users()
        assert len(users) == 3

    def test_step_updates_streams(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, seed=42)
        adapter.step(0)
        for stream in adapter.get_streams():
            assert stream.current_data.size == stream.dimensionality

    def test_multiple_steps(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, seed=42)
        for step in range(50):
            adapter.step(step)

    def test_ground_truth_bool(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, seed=42)
        adapter.step(0)
        result = adapter.get_ground_truth(0)
        assert isinstance(result, bool)

    def test_score_relevance(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10)
        users = adapter.get_users()
        signal = np.ones(10)
        score = adapter.score_relevance(signal, users[0])
        assert isinstance(score, float)

    def test_compute_costs(self) -> None:
        adapter = FireEcologyAdapter()
        costs = adapter.compute_costs(n_escalations=5, n_correct=3, n_false_alarms=1, n_missed=2)
        assert "surveillance_cost" in costs
        assert "response_cost" in costs
        assert "damage_cost" in costs
        assert costs["damage_cost"] > costs["response_cost"]

    def test_fire_grid_accessible(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10)
        assert adapter.fire_grid.rows == 10

    def test_default_thermal_dim_capped(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=30, grid_cols=30)
        thermal = adapter.get_streams()[0]
        assert thermal.dimensionality == FireEcologyAdapter.DEFAULT_THERMAL_DIM

    def test_custom_thermal_dim(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, max_thermal_dim=50)
        thermal = adapter.get_streams()[0]
        # 10*10=100 cells, capped to 50
        assert thermal.dimensionality == 50

    def test_thermal_dim_uses_full_grid_when_smaller(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=3, grid_cols=3, max_thermal_dim=100)
        thermal = adapter.get_streams()[0]
        # 3*3=9 cells, smaller than cap
        assert thermal.dimensionality == 9

    def test_thermal_dim_none_uses_default(self) -> None:
        a1 = FireEcologyAdapter(grid_rows=10, grid_cols=10, max_thermal_dim=None)
        a2 = FireEcologyAdapter(grid_rows=10, grid_cols=10)
        assert a1.get_streams()[0].dimensionality == a2.get_streams()[0].dimensionality

    def test_weather_evolves(self) -> None:
        adapter = FireEcologyAdapter(seed=42)
        temps: list[float] = []
        for step in range(20):
            adapter.step(step)
            temps.append(adapter.weather.temperature)
        assert len(set(round(t, 2) for t in temps)) > 1
