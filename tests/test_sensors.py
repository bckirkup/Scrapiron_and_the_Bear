"""Tests for sensor models."""

from __future__ import annotations

import numpy as np

from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.fuel import FuelCell
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.camera_tower import CameraTower
from fire_ecology.sensors.fuel_moisture import FuelMoistureSensor
from fire_ecology.sensors.opir import OPIRSatellite
from fire_ecology.sensors.weather_station import WeatherStation


class TestOPIRSatellite:
    def test_scan_on_cadence(self) -> None:
        rng = np.random.default_rng(42)
        opir = OPIRSatellite(cadence=5)
        grid = FireGrid(rows=10, cols=10)
        grid.ignite(5, 5, 0)
        result = opir.scan(grid, time_step=5, rng=rng)
        assert len(result) > 0

    def test_scan_off_cadence_empty(self) -> None:
        rng = np.random.default_rng(42)
        opir = OPIRSatellite(cadence=5)
        grid = FireGrid(rows=10, cols=10)
        grid.ignite(5, 5, 0)
        result = opir.scan(grid, time_step=3, rng=rng)
        assert len(result) == 0

    def test_scan_no_fires(self) -> None:
        rng = np.random.default_rng(42)
        opir = OPIRSatellite(cadence=1, false_positive_rate=0.0)
        grid = FireGrid(rows=5, cols=5)
        result = opir.scan(grid, time_step=0, rng=rng)
        assert len(result) == 0


class TestCameraTower:
    def test_detect_nearby_fire(self) -> None:
        rng = np.random.default_rng(42)
        camera = CameraTower(row=5, col=5, max_range=10.0, detection_probability=1.0)
        grid = FireGrid(rows=10, cols=10)
        grid.ignite(6, 6, 0)
        detections = camera.detect(grid, is_night=False, rng=rng)
        assert len(detections) > 0

    def test_detect_out_of_range(self) -> None:
        rng = np.random.default_rng(42)
        camera = CameraTower(row=0, col=0, max_range=3.0)
        grid = FireGrid(rows=20, cols=20)
        grid.ignite(19, 19, 0)
        detections = camera.detect(grid, is_night=False, rng=rng)
        assert len(detections) == 0

    def test_night_penalty_reduces_detection(self) -> None:
        rng_day = np.random.default_rng(42)
        rng_night = np.random.default_rng(42)
        camera = CameraTower(row=5, col=5, max_range=10.0, detection_probability=0.9)
        grid = FireGrid(rows=10, cols=10)
        for r in range(3, 8):
            for c in range(3, 8):
                grid.ignite(r, c, 0)
        day = camera.detect(grid, is_night=False, rng=rng_day)
        grid2 = FireGrid(rows=10, cols=10)
        for r in range(3, 8):
            for c in range(3, 8):
                grid2.ignite(r, c, 0)
        night = camera.detect(grid2, is_night=True, rng=rng_night)
        assert len(night) <= len(day)


class TestWeatherStation:
    def test_observe_shape(self) -> None:
        rng = np.random.default_rng(42)
        station = WeatherStation(row=0, col=0)
        weather = WeatherState()
        obs = station.observe(weather, rng)
        assert obs.shape == (5,)

    def test_observe_nonneg_values(self) -> None:
        rng = np.random.default_rng(42)
        station = WeatherStation(row=0, col=0)
        weather = WeatherState()
        obs = station.observe(weather, rng)
        assert obs[1] >= 0.0
        assert obs[2] >= 0.0
        assert obs[4] >= 0.0


class TestFuelMoistureSensor:
    def test_on_cadence_returns_data(self) -> None:
        rng = np.random.default_rng(42)
        sensor = FuelMoistureSensor(row=0, col=0, cadence=5)
        fuel = FuelCell()
        obs = sensor.observe(fuel, time_step=5, rng=rng)
        assert obs is not None
        assert obs.shape == (3,)

    def test_off_cadence_returns_none(self) -> None:
        rng = np.random.default_rng(42)
        sensor = FuelMoistureSensor(row=0, col=0, cadence=5)
        fuel = FuelCell()
        obs = sensor.observe(fuel, time_step=3, rng=rng)
        assert obs is None

    def test_values_bounded(self) -> None:
        rng = np.random.default_rng(42)
        sensor = FuelMoistureSensor(row=0, col=0, cadence=1)
        fuel = FuelCell()
        obs = sensor.observe(fuel, time_step=0, rng=rng)
        assert obs is not None
        assert all(0.0 <= v <= 1.0 for v in obs)
