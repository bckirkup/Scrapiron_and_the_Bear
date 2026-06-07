"""Tests for fire metrics tracking."""

from __future__ import annotations

import pytest

from fire_ecology.metrics.fire_metrics import FireMetrics, StepMetrics


class TestStepMetrics:
    def test_default_values(self) -> None:
        m = StepMetrics(time_step=0)
        assert m.active_fires == 0
        assert m.burned_area == 0

    def test_custom_values(self) -> None:
        m = StepMetrics(time_step=5, active_fires=3, burned_area=10, detections=2)
        assert m.active_fires == 3
        assert m.detections == 2


class TestFireMetrics:
    def test_record_step(self) -> None:
        metrics = FireMetrics()
        metrics.record_step(StepMetrics(time_step=0, detections=5))
        assert len(metrics.history) == 1
        assert metrics.total_fires_detected == 5

    def test_mean_detection_latency_empty(self) -> None:
        metrics = FireMetrics()
        assert metrics.mean_detection_latency == float("inf")

    def test_mean_detection_latency(self) -> None:
        metrics = FireMetrics()
        metrics.record_detection_latency(2)
        metrics.record_detection_latency(4)
        assert metrics.mean_detection_latency == pytest.approx(3.0)

    def test_suppression_success_rate(self) -> None:
        metrics = FireMetrics()
        metrics.total_suppressions_attempted = 10
        metrics.total_suppressions_successful = 7
        assert metrics.suppression_success_rate == pytest.approx(0.7)

    def test_false_dispatch_rate(self) -> None:
        metrics = FireMetrics()
        metrics.total_fires_detected = 8
        metrics.total_false_dispatches = 2
        assert metrics.false_dispatch_rate == pytest.approx(0.2)

    def test_opir_rescue_rate(self) -> None:
        metrics = FireMetrics()
        metrics.total_fires_detected = 10
        metrics.total_opir_rescues = 3
        assert metrics.opir_rescue_rate == pytest.approx(0.3)

    def test_total_cost(self) -> None:
        metrics = FireMetrics()
        metrics.record_step(
            StepMetrics(time_step=0, surveillance_cost=1.0, response_cost=2.0, damage_cost=3.0)
        )
        metrics.record_step(
            StepMetrics(time_step=1, surveillance_cost=0.5, response_cost=1.0, damage_cost=0.0)
        )
        assert metrics.total_cost == pytest.approx(7.5)

    def test_summary_keys(self) -> None:
        metrics = FireMetrics()
        metrics.record_step(StepMetrics(time_step=0))
        summary = metrics.summary()
        assert "mean_detection_latency" in summary
        assert "total_cost" in summary
        assert "opir_rescue_rate" in summary

    def test_record_detections_from_grid(self) -> None:
        from fire_ecology.environment.fire import FireGrid

        grid = FireGrid(rows=5, cols=5)
        grid.ignite(2, 2, time_step=3)

        metrics = FireMetrics()
        metrics.record_detections_from_grid([(2, 2)], grid, time_step=5)
        assert len(metrics.detection_latencies) == 1
        assert metrics.detection_latencies[0] == 2  # 5 - 3 = 2

    def test_record_detections_from_grid_deduplication(self) -> None:
        from fire_ecology.environment.fire import FireGrid

        grid = FireGrid(rows=5, cols=5)
        grid.ignite(1, 1, time_step=0)

        metrics = FireMetrics()
        metrics.record_detections_from_grid([(1, 1)], grid, time_step=2)
        metrics.record_detections_from_grid([(1, 1)], grid, time_step=3)
        assert len(metrics.detection_latencies) == 1
