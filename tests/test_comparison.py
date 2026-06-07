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

    def test_detection_latency_not_inf(self) -> None:
        """Detection latency must be finite when fires are detected."""
        config = ComparisonConfig(
            steps=30,
            grid_rows=10,
            grid_cols=10,
            seed=42,
            n_drones=5,
            include_a4=False,
        )
        results = run_comparison(config)
        for r in results:
            if r.detections > 0:
                assert r.mean_detection_latency < float("inf"), (
                    f"{r.name}: latency is inf despite {r.detections} detections"
                )

    def test_detection_latency_not_inf_with_a4(self) -> None:
        """A4/BMA also produces finite detection latency."""
        config = ComparisonConfig(
            steps=30,
            grid_rows=10,
            grid_cols=10,
            seed=42,
            n_drones=5,
            include_a4=True,
        )
        results = run_comparison(config)
        a4 = [r for r in results if r.name == "A4 BMA"][0]
        if a4.detections > 0:
            assert a4.mean_detection_latency < float("inf")

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
