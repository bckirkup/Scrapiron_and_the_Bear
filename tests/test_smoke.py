"""Smoke tests: end-to-end simulation validates emergent behavior."""

from __future__ import annotations

import pytest

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.metrics.fire_metrics import FireMetrics, StepMetrics


@pytest.mark.smoke
class TestSmoke:
    def test_full_simulation_runs(self) -> None:
        """Verify a full simulation runs without errors."""
        adapter = FireEcologyAdapter(grid_rows=15, grid_cols=15, n_cameras=3, seed=42)
        metrics = FireMetrics()

        for step in range(100):
            adapter.step(step)
            active = adapter.fire_grid.active_fire_cells()
            step_metrics = StepMetrics(
                time_step=step,
                active_fires=len(active),
                burned_area=adapter.fire_grid.burned_area(),
            )
            metrics.record_step(step_metrics)

        assert len(metrics.history) == 100

    def test_fire_eventually_ignites(self) -> None:
        """Over enough steps with dry weather, at least one fire should start."""
        adapter = FireEcologyAdapter(grid_rows=30, grid_cols=30, seed=123)
        any_fire = False
        for step in range(200):
            adapter.step(step)
            if adapter.fire_grid.active_fire_cells():
                any_fire = True
                break
        assert any_fire

    def test_streams_consistent_dimensions(self) -> None:
        """Stream dimensions should remain constant across all steps."""
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, seed=42)
        expected_dims = [s.dimensionality for s in adapter.get_streams()]
        for step in range(50):
            adapter.step(step)
            actual_dims = [s.current_data.shape[0] for s in adapter.get_streams()]
            assert actual_dims == expected_dims

    def test_weather_varies(self) -> None:
        """Weather should change across steps."""
        adapter = FireEcologyAdapter(seed=42)
        temps = set[float]()
        for step in range(30):
            adapter.step(step)
            temps.add(round(adapter.weather.temperature, 1))
        assert len(temps) > 5
