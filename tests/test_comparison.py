"""Tests for head-to-head architecture comparison harness."""

from __future__ import annotations

from fire_ecology.comparison import (
    ComparisonConfig,
    format_comparison_json,
    format_comparison_table,
    run_comparison,
)


class TestComparison:
    def test_run_without_a4(self) -> None:
        config = ComparisonConfig(
            steps=10,
            grid_rows=5,
            grid_cols=5,
            seed=42,
            n_drones=3,
            include_a4=False,
        )
        results = run_comparison(config)
        assert len(results) == 4
        names = [r.name for r in results]
        assert "A0 Human" in names
        assert "A4 BMA" not in names

    def test_run_with_a4(self) -> None:
        config = ComparisonConfig(
            steps=10,
            grid_rows=5,
            grid_cols=5,
            seed=42,
            n_drones=3,
            include_a4=True,
        )
        results = run_comparison(config)
        assert len(results) == 5
        assert results[-1].name == "A4 BMA"

    def test_deterministic(self) -> None:
        config = ComparisonConfig(
            steps=15,
            grid_rows=5,
            grid_cols=5,
            seed=99,
            n_drones=3,
            include_a4=False,
        )
        r1 = run_comparison(config)
        r2 = run_comparison(config)
        for a, b in zip(r1, r2, strict=True):
            assert a.detections == b.detections
            assert a.burned_cells == b.burned_cells
            assert a.cost == b.cost

    def test_format_table(self) -> None:
        config = ComparisonConfig(
            steps=5,
            grid_rows=5,
            grid_cols=5,
            seed=42,
            n_drones=2,
            include_a4=False,
        )
        results = run_comparison(config)
        table = format_comparison_table(results)
        assert "Architecture" in table
        assert "A0 Human" in table

    def test_format_json(self) -> None:
        config = ComparisonConfig(
            steps=5,
            grid_rows=5,
            grid_cols=5,
            seed=42,
            n_drones=2,
            include_a4=False,
        )
        results = run_comparison(config)
        j = format_comparison_json(results)
        assert '"architecture"' in j
        assert '"detections"' in j
