"""Tests for the FireEcology domain adapter."""

from __future__ import annotations

import numpy as np
from tattletots.models.dispatch_target import DispatchTarget
from tattletots.models.observation import ObservationStatus
from tattletots.models.report import Report

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState


class TestFireEcologyAdapter:
    def test_get_streams(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10)
        streams = adapter.get_streams()
        assert len(streams) == 3
        assert all(s.dimensionality > 0 for s in streams)

    def test_streams_publish_spatial_contract(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=20, grid_cols=20)
        thermal, weather, fuel = adapter.get_streams()

        assert adapter.get_location_frame() == ((0, 0), (19, 19))
        assert thermal.metadata is not None
        assert thermal.metadata.coordinates is not None
        assert len(thermal.metadata.coordinates) == thermal.dimensionality
        assert thermal.metadata.footprints is not None
        assert thermal.metadata.footprints[0] is not None
        assert thermal.metadata.footprints[0][0] > 1.0

        assert weather.metadata is not None
        assert weather.metadata.sensor_coordinates is not None
        assert weather.metadata.modality[:5] == [
            "temperature",
            "humidity",
            "wind_speed",
            "wind_direction",
            "precipitation",
        ]
        assert fuel.metadata is not None
        assert fuel.metadata.sensor_coordinates is not None
        assert fuel.metadata.modality[:3] == [
            "live_moisture",
            "dead_moisture",
            "effective_moisture",
        ]

    def test_sensor_declarations_do_not_depend_on_fire_state(self) -> None:
        quiet_grid = FireGrid(rows=20, cols=20)
        burning_grid = FireGrid(rows=20, cols=20)
        for row, col in ((2, 3), (8, 11), (17, 19)):
            burning_grid.ignite(row, col, 0)
        quiet = FireEcologyAdapter(grid_rows=20, grid_cols=20, seed=42)
        burning = FireEcologyAdapter(grid_rows=20, grid_cols=20, seed=42)
        weather = WeatherState(temperature=30.0, humidity=0.3, wind_speed=5.0)

        quiet.observe_grid(quiet_grid, weather, time_step=1)
        burning.observe_grid(burning_grid, weather, time_step=1)

        for quiet_stream, burning_stream in zip(
            quiet.get_streams(), burning.get_streams(), strict=True
        ):
            assert quiet_stream.metadata == burning_stream.metadata
            np.testing.assert_array_equal(
                quiet_stream.current_status, burning_stream.current_status
            )

    def test_thermal_sensor_coordinates_are_static_and_populated(self) -> None:
        quiet_grid = FireGrid(rows=20, cols=20)
        burning_grid = FireGrid(rows=20, cols=20)
        burning_grid.ignite(10, 10, 0)
        quiet = FireEcologyAdapter(grid_rows=20, grid_cols=20, seed=42)
        burning = FireEcologyAdapter(grid_rows=20, grid_cols=20, seed=42)
        weather = WeatherState(temperature=30.0, humidity=0.3, wind_speed=5.0)

        initial = quiet.get_streams()[0].metadata
        assert initial is not None
        assert initial.sensor_coordinates is not None
        assert all(coordinate is not None for coordinate in initial.sensor_coordinates)

        for time_step in (1, 5):
            quiet.observe_grid(quiet_grid, weather, time_step=time_step)
            burning.observe_grid(burning_grid, weather, time_step=time_step)
            quiet_metadata = quiet.get_streams()[0].metadata
            burning_metadata = burning.get_streams()[0].metadata
            assert quiet_metadata is not None
            assert burning_metadata is not None
            assert quiet_metadata.sensor_coordinates == initial.sensor_coordinates
            assert burning_metadata.sensor_coordinates == initial.sensor_coordinates

    def test_statuses_use_sensor_availability_not_fire_state(self) -> None:
        quiet_grid = FireGrid(rows=20, cols=20)
        burning_grid = FireGrid(rows=20, cols=20)
        burning_grid.ignite(10, 10, 0)
        quiet = FireEcologyAdapter(grid_rows=20, grid_cols=20, seed=7, opir_cadence=5)
        burning = FireEcologyAdapter(grid_rows=20, grid_cols=20, seed=7, opir_cadence=5)
        weather = WeatherState(temperature=30.0, humidity=0.3, wind_speed=5.0)

        quiet.observe_grid(quiet_grid, weather, time_step=2)
        burning.observe_grid(burning_grid, weather, time_step=2)

        for quiet_stream, burning_stream in zip(
            quiet.get_streams(), burning.get_streams(), strict=True
        ):
            assert quiet_stream.current_status.size == quiet_stream.dimensionality
            np.testing.assert_array_equal(
                quiet_stream.current_status, burning_stream.current_status
            )
        assert ObservationStatus.MISSING.value in quiet.get_streams()[2].current_status

    def test_thermal_stream_uses_sensor_returns_not_grid_truth(self) -> None:
        quiet_grid = FireGrid(rows=10, cols=10)
        burning_grid = FireGrid(rows=10, cols=10)
        burning_grid.ignite(5, 5, 0)
        quiet = FireEcologyAdapter(
            grid_rows=10,
            grid_cols=10,
            n_cameras=0,
            opir_cadence=1,
            seed=42,
        )
        burning = FireEcologyAdapter(
            grid_rows=10,
            grid_cols=10,
            n_cameras=0,
            opir_cadence=1,
            seed=42,
        )
        quiet.opir.miss_rate = 1.0
        burning.opir.miss_rate = 1.0
        weather = WeatherState(temperature=30.0, humidity=0.3, wind_speed=5.0)

        quiet.observe_grid(quiet_grid, weather, time_step=0)
        burning.observe_grid(burning_grid, weather, time_step=0)

        thermal = burning.get_streams()[0]
        assert np.max(thermal.current_data) == 0.0
        np.testing.assert_array_equal(
            quiet.get_streams()[0].current_status,
            burning.get_streams()[0].current_status,
        )
        assert np.all(burning.get_streams()[0].current_status == ObservationStatus.OBSERVED.value)

    def test_night_changes_detection_not_coverage_status(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=20, grid_cols=20, opir_cadence=5, seed=42)

        day_status = adapter._thermal_status(6, adapter.fire_grid)
        night_status = adapter._thermal_status(18, adapter.fire_grid)

        np.testing.assert_array_equal(day_status, night_status)

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
        assert len({round(t, 2) for t in temps}) > 1

    def test_base_ignition_rate_increases_burned_area(self) -> None:
        low = FireEcologyAdapter(
            grid_rows=10, grid_cols=10, seed=42, base_ignition_rate=0.00001, n_drones=0
        )
        high = FireEcologyAdapter(
            grid_rows=10, grid_cols=10, seed=42, base_ignition_rate=0.01, n_drones=0
        )
        for step in range(100):
            low.step(step)
            high.step(step)
        assert high.fire_grid.burned_area() >= low.fire_grid.burned_area()

    def test_drone_suppression_reduces_active_fires(self) -> None:
        adapter = FireEcologyAdapter(
            grid_rows=10, grid_cols=10, seed=7, base_ignition_rate=0.05, n_drones=5
        )
        for step in range(20):
            adapter.step(step)
        before = len(adapter.fire_grid.active_fire_cells())
        if before == 0:
            adapter.fire_grid.ignite(5, 5, 0)
            before = len(adapter.fire_grid.active_fire_cells())
        suppressed = adapter.dispatch_drone_suppression(before)
        assert suppressed >= 0
        assert len(adapter.fire_grid.active_fire_cells()) <= before

    def test_observe_grid_reads_external_without_stepping_internal(self) -> None:
        external = FireGrid(rows=10, cols=10)
        external.ignite(5, 5, 0)
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, seed=42, max_thermal_dim=100)
        weather = WeatherState(temperature=30.0, humidity=0.3, wind_speed=5.0)

        adapter.observe_grid(external, weather, time_step=0)

        assert len(adapter.fire_grid.active_fire_cells()) == 0
        thermal = adapter.get_streams()[0].current_data
        assert float(np.max(thermal)) > 0.0

    def test_thermal_location_inverts_sample_index(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, max_thermal_dim=5)
        adapter.opir.miss_rate = 0.0
        adapter.opir.false_positive_rate = 0.0
        external = FireGrid(rows=10, cols=10)
        external.ignite(4, 9, 0)
        adapter.observe_grid(
            external,
            WeatherState(temperature=30.0, humidity=0.3, wind_speed=5.0),
            time_step=0,
        )
        thermal = adapter.get_streams()[0].current_data

        location = adapter.infer_report_location(
            [thermal],
            ["thermal_detection"],
        )

        assert location == (4, 9)
        assert adapter._thermal_sample_indices(5).tolist() == [0, 24, 49, 74, 99]

    def test_thermal_location_resolution_excludes_unsampled_cells(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, max_thermal_dim=5)
        sampled = set(adapter._thermal_sample_indices(5).tolist())
        unsampled_index = next(index for index in range(100) if index not in sampled)
        external = FireGrid(rows=10, cols=10)
        external.ignite(unsampled_index // 10, unsampled_index % 10, 0)

        adapter.observe_grid(
            external,
            WeatherState(temperature=30.0, humidity=0.3, wind_speed=5.0),
            time_step=0,
        )

        thermal = adapter.get_streams()[0].current_data
        assert float(np.max(thermal)) == 0.0
        assert unsampled_index not in sampled

    def test_empty_thermal_vector_uses_first_declared_sample(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, max_thermal_dim=5)

        location = adapter.infer_report_location(
            [np.zeros(5)],
            ["thermal_detection"],
        )

        first_index = int(adapter._thermal_sample_indices(5)[0])
        assert location == (first_index // 10, first_index % 10)

    def test_dispatch_and_judge_necessary_suppression(self) -> None:
        adapter = FireEcologyAdapter(
            grid_rows=10,
            grid_cols=10,
            n_drones=1,
            suppression_effectiveness=0.99,
        )
        adapter.fire_grid.ignite(5, 5, 0)
        users = adapter.get_users()
        report = Report(
            agent_id="agent-1",
            target_user_id=users[1].id,
            time_step=0,
            signal_vector=np.ones(10),
            confidence=0.9,
            anomaly_score=2.0,
            location=(5, 5),
            verified=True,
            correct=True,
        )
        outcomes = adapter.dispatch_and_judge_responses(
            [
                DispatchTarget(
                    location=(5, 5),
                    reports=[report],
                    responder_user_id=adapter.get_responder_user_id(),
                    cop_threat_level=2.0,
                )
            ],
            0,
        )
        assert len(outcomes) == 1
        assert outcomes[0].dispatched
        assert outcomes[0].response_necessary

    def test_dispatch_and_judge_unnecessary_without_fire(self) -> None:
        adapter = FireEcologyAdapter(grid_rows=10, grid_cols=10, n_drones=1)
        users = adapter.get_users()
        report = Report(
            agent_id="agent-1",
            target_user_id=users[1].id,
            time_step=0,
            signal_vector=np.ones(10),
            confidence=0.9,
            anomaly_score=2.0,
            location=(2, 2),
            verified=True,
            correct=True,
        )
        outcomes = adapter.dispatch_and_judge_responses(
            [
                DispatchTarget(
                    location=(2, 2),
                    reports=[report],
                    responder_user_id=adapter.get_responder_user_id(),
                    cop_threat_level=2.0,
                )
            ],
            0,
        )
        assert len(outcomes) == 1
        assert outcomes[0].dispatched
        assert not outcomes[0].response_necessary

    def test_dispatch_not_gated_by_report_correct(self) -> None:
        adapter = FireEcologyAdapter(
            grid_rows=10,
            grid_cols=10,
            n_drones=1,
            suppression_effectiveness=0.99,
        )
        adapter.fire_grid.ignite(4, 4, 0)
        users = adapter.get_users()
        report = Report(
            agent_id="agent-1",
            target_user_id=users[1].id,
            time_step=0,
            signal_vector=np.ones(10),
            confidence=0.9,
            anomaly_score=2.0,
            location=(4, 4),
            verified=True,
            correct=False,
        )
        outcomes = adapter.dispatch_and_judge_responses(
            [
                DispatchTarget(
                    location=(4, 4),
                    reports=[report],
                    responder_user_id=adapter.get_responder_user_id(),
                    cop_threat_level=2.0,
                )
            ],
            0,
        )
        assert outcomes[0].dispatched
        assert outcomes[0].response_necessary
