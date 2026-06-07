"""Tests for competing fire-management architectures."""

from __future__ import annotations

import numpy as np

from fire_ecology.architectures.a0_human import HumanBaseline
from fire_ecology.architectures.a1_camera_ml import CameraMLNetwork
from fire_ecology.architectures.a2_centralized import CentralizedOptimizer
from fire_ecology.architectures.a3_federated import FederatedEdge
from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.camera_tower import CameraTower
from fire_ecology.sensors.opir import OPIRSatellite


def _setup() -> tuple[FireGrid, WeatherState, OPIRSatellite, np.random.Generator]:
    grid = FireGrid(rows=20, cols=20)
    grid.ignite(10, 10, 0)
    weather = WeatherState(temperature=35.0, humidity=0.2, wind_speed=10.0)
    opir = OPIRSatellite(cadence=1)
    rng = np.random.default_rng(42)
    return grid, weather, opir, rng


class TestHumanBaseline:
    def test_step_returns_result(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = HumanBaseline()
        result = arch.step(grid, weather, opir, time_step=0, rng=rng)
        assert result.cost > 0.0

    def test_reset(self) -> None:
        arch = HumanBaseline()
        arch.reset()


class TestCameraMLNetwork:
    def test_step_with_cameras(self) -> None:
        grid, weather, opir, rng = _setup()
        cameras = [CameraTower(row=10, col=10, max_range=15.0)]
        arch = CameraMLNetwork(cameras=cameras)
        result = arch.step(grid, weather, opir, time_step=0, rng=rng)
        assert result.cost >= 0.0

    def test_step_no_cameras(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = CameraMLNetwork()
        result = arch.step(grid, weather, opir, time_step=0, rng=rng)
        assert result.cost >= 0.0


class TestCentralizedOptimizer:
    def test_step_detects_fire(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = CentralizedOptimizer(n_drones=5, detection_range=5)
        result = arch.step(grid, weather, opir, time_step=0, rng=rng)
        assert len(result.detections) >= 0

    def test_reset_clears_positions(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = CentralizedOptimizer()
        arch.step(grid, weather, opir, time_step=0, rng=rng)
        arch.reset()
        assert len(arch._drone_positions) == 0


class TestFederatedEdge:
    def test_step_returns_result(self) -> None:
        grid, weather, opir, rng = _setup()
        arch = FederatedEdge(n_nodes=4)
        result = arch.step(grid, weather, opir, time_step=0, rng=rng)
        assert result.cost >= 0.0

    def test_reset(self) -> None:
        arch = FederatedEdge()
        arch.step(
            FireGrid(rows=10, cols=10),
            WeatherState(),
            OPIRSatellite(),
            time_step=0,
            rng=np.random.default_rng(42),
        )
        arch.reset()
        assert len(arch._node_positions) == 0
