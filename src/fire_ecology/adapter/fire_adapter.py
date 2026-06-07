"""FireEcology domain adapter: connects fire simulation to TattleTots engine.

Implements the DomainAdapter ABC so the TattleTots engine can drive a fire
ecology simulation without any domain-specific knowledge.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User

from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.camera_tower import CameraTower
from fire_ecology.sensors.fuel_moisture import FuelMoistureSensor
from fire_ecology.sensors.opir import OPIRSatellite
from fire_ecology.sensors.weather_station import WeatherStation
from fire_ecology.users.fire_users import create_fire_users


class FireEcologyAdapter(DomainAdapter):  # type: ignore[misc]
    """Domain adapter bridging fire ecology simulation to TattleTots.

    Manages the fire grid, weather, sensors, and translates their outputs
    into abstract data streams consumable by Tot agents.
    """

    def __init__(
        self,
        grid_rows: int = 20,
        grid_cols: int = 20,
        n_cameras: int = 3,
        n_weather_stations: int = 4,
        n_fuel_sensors: int = 2,
        opir_cadence: int = 5,
        seed: int = 42,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self._grid = FireGrid(rows=grid_rows, cols=grid_cols)
        self._weather = WeatherState()
        self._opir = OPIRSatellite(cadence=opir_cadence)

        self._cameras = self._place_cameras(n_cameras)
        self._weather_stations = self._place_weather_stations(n_weather_stations)
        self._fuel_sensors = self._place_fuel_sensors(n_fuel_sensors)

        self._streams: list[Stream] = []
        self._users: list[User] = []
        self._current_step = 0
        self._setup_streams()
        self._users = create_fire_users(n_signal_dims=self._total_stream_dims())

    def _place_cameras(self, n: int) -> list[CameraTower]:
        """Place camera towers on ridge positions (higher elevation cells)."""
        cameras: list[CameraTower] = []
        for i in range(n):
            r = int((i + 1) * self._grid.rows / (n + 1))
            c = int((i + 1) * self._grid.cols / (n + 1))
            cameras.append(CameraTower(row=r, col=c, max_range=10.0))
        return cameras

    def _place_weather_stations(self, n: int) -> list[WeatherStation]:
        stations: list[WeatherStation] = []
        for _ in range(n):
            r = int(self.rng.integers(0, self._grid.rows))
            c = int(self.rng.integers(0, self._grid.cols))
            stations.append(WeatherStation(row=r, col=c))
        return stations

    def _place_fuel_sensors(self, n: int) -> list[FuelMoistureSensor]:
        sensors: list[FuelMoistureSensor] = []
        for _ in range(n):
            r = int(self.rng.integers(0, self._grid.rows))
            c = int(self.rng.integers(0, self._grid.cols))
            sensors.append(FuelMoistureSensor(row=r, col=c))
        return sensors

    def _setup_streams(self) -> None:
        """Create data streams from sensor outputs.

        Stream layout:
        - thermal_stream: OPIR + camera thermal detections (grid_rows * grid_cols dims → capped)
        - weather_stream: weather station readings (5 dims per station)
        - fuel_stream: fuel moisture readings (3 dims per sensor)
        """
        thermal_dim = min(self._grid.rows * self._grid.cols, 30)
        weather_dim = len(self._weather_stations) * 5
        fuel_dim = len(self._fuel_sensors) * 3

        self._streams = [
            Stream(
                stream_type=StreamType.RAW,
                dimensionality=thermal_dim,
                label="thermal_detection",
                current_data=np.zeros(thermal_dim),
            ),
            Stream(
                stream_type=StreamType.RAW,
                dimensionality=weather_dim,
                label="weather_observations",
                current_data=np.zeros(weather_dim),
            ),
            Stream(
                stream_type=StreamType.RAW,
                dimensionality=fuel_dim,
                label="fuel_moisture",
                current_data=np.zeros(fuel_dim),
            ),
        ]

    def _total_stream_dims(self) -> int:
        return sum(s.dimensionality for s in self._streams)

    def get_streams(self) -> list[Stream]:
        return self._streams

    def get_users(self) -> list[User]:
        return self._users

    def step(self, time_step: int) -> None:
        """Advance fire simulation and update all sensor streams."""
        self._current_step = time_step
        self._evolve_weather(time_step)
        self._grid.stochastic_ignition(self._weather, time_step, self.rng)
        self._grid.step(self._weather, time_step, self.rng)
        self._update_fuel_moisture()
        self._update_streams(time_step)

    def get_ground_truth(self, time_step: int) -> bool:
        """A fire event is active if any cells are currently burning."""
        return len(self._grid.active_fire_cells()) > 0

    def score_relevance(self, signal_vector: NDArray[np.float64], user: User) -> float:
        return float(user.compute_relevance(signal_vector))

    def compute_costs(
        self,
        n_escalations: int,
        n_correct: int,
        n_false_alarms: int,
        n_missed: int,
    ) -> dict[str, float]:
        """Fire-domain cost model.

        - Surveillance cost: proportional to sensor/drone operations.
        - Response cost: dispatch and suppression expenses.
        - Damage cost: uncontrolled fire damage (missed detections are expensive).
        """
        return {
            "surveillance_cost": n_escalations * 0.5,
            "response_cost": n_correct * 2.0 + n_false_alarms * 1.0,
            "damage_cost": n_missed * 10.0,
        }

    def _evolve_weather(self, time_step: int) -> None:
        """Simple sinusoidal weather with noise."""
        phase = 2.0 * np.pi * time_step / 200.0
        self._weather = WeatherState(
            temperature=25.0 + 10.0 * np.sin(phase) + float(self.rng.normal(0, 2)),
            humidity=float(np.clip(0.4 + 0.2 * np.cos(phase) + self.rng.normal(0, 0.05), 0, 1)),
            wind_speed=max(0.0, 5.0 + 3.0 * np.sin(phase * 0.7) + float(self.rng.normal(0, 1))),
            wind_direction=float((180.0 + 90.0 * np.sin(phase * 0.3)) % 360),
            precipitation=max(
                0.0, float(self.rng.exponential(0.5) if self.rng.random() < 0.1 else 0.0)
            ),
        )

    def _update_fuel_moisture(self) -> None:
        """Dry out or wet fuel based on weather."""
        drying = self._weather.moisture_drying_rate()
        wetting = min(self._weather.precipitation * 0.01, 0.1)
        for r in range(self._grid.rows):
            for c in range(self._grid.cols):
                fuel = self._grid.fuel[r][c]
                fuel.dead_moisture = float(
                    np.clip(fuel.dead_moisture - drying + wetting, 0.05, 1.0)
                )
                fuel.live_moisture = float(
                    np.clip(fuel.live_moisture - drying * 0.5 + wetting * 0.7, 0.1, 1.0)
                )

    def _update_streams(self, time_step: int) -> None:
        """Populate stream data from sensor outputs."""
        thermal = self._build_thermal_vector(time_step)
        self._streams[0].update(thermal)

        weather_obs = np.concatenate(
            [ws.observe(self._weather, self.rng) for ws in self._weather_stations]
        )
        self._streams[1].update(weather_obs)

        fuel_parts: list[np.ndarray] = []
        for fs in self._fuel_sensors:
            obs = fs.observe(self._grid.fuel[fs.row][fs.col], time_step, self.rng)
            fuel_parts.append(obs if obs is not None else np.zeros(3))
        self._streams[2].update(np.concatenate(fuel_parts))

    def _build_thermal_vector(self, time_step: int) -> np.ndarray:
        """Flatten fire intensity grid into a capped-dimension stream."""
        dim = self._streams[0].dimensionality
        full_grid = np.zeros(self._grid.rows * self._grid.cols)
        for r in range(self._grid.rows):
            for c in range(self._grid.cols):
                fs = self._grid.fire[r][c]
                full_grid[r * self._grid.cols + c] = fs.intensity if fs.is_active else 0.0

        if full_grid.size <= dim:
            result = np.zeros(dim)
            result[: full_grid.size] = full_grid
            return result

        indices = np.linspace(0, full_grid.size - 1, dim, dtype=int)
        return full_grid[indices]

    @property
    def fire_grid(self) -> FireGrid:
        """Expose fire grid for external inspection (metrics, architectures)."""
        return self._grid

    @property
    def weather(self) -> WeatherState:
        """Expose current weather state."""
        return self._weather

    @property
    def opir(self) -> OPIRSatellite:
        """Expose OPIR satellite sensor."""
        return self._opir
