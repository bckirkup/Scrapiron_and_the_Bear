"""FireEcology domain adapter: connects fire simulation to TattleTots engine.

Implements the DomainAdapter ABC so the TattleTots engine can drive a fire
ecology simulation without any domain-specific knowledge.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from tattletots.engine.response_judgment import judge_necessity
from tattletots.interface.domain_adapter import DomainAdapter
from tattletots.models.dispatch_target import DispatchTarget
from tattletots.models.location import EventLocation
from tattletots.models.observation import ObservationStatus, StreamMetadata
from tattletots.models.report import Report
from tattletots.models.response_outcome import ResponseOutcome
from tattletots.models.stream import Stream, StreamType
from tattletots.models.user import User

from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.camera_tower import CameraTower, is_night_time
from fire_ecology.sensors.fuel_moisture import FuelMoistureSensor
from fire_ecology.sensors.opir import OPIRSatellite
from fire_ecology.sensors.weather_station import WeatherStation
from fire_ecology.users.fire_users import create_fire_users


class FireEcologyAdapter(DomainAdapter):
    """Domain adapter bridging fire ecology simulation to TattleTots.

    Manages the fire grid, weather, sensors, and translates their outputs
    into abstract data streams consumable by Tot agents.
    """

    DEFAULT_THERMAL_DIM: int = 30

    def __init__(
        self,
        grid_rows: int = 20,
        grid_cols: int = 20,
        n_cameras: int = 3,
        n_weather_stations: int = 4,
        n_fuel_sensors: int = 2,
        opir_cadence: int = 5,
        use_opir: bool = True,
        seed: int = 42,
        max_thermal_dim: int | None = None,
        base_ignition_rate: float = 0.0001,
        weather_volatility: float = 1.0,
        n_drones: int = 0,
        suppression_effectiveness: float = 0.99,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self._grid = FireGrid(rows=grid_rows, cols=grid_cols)
        self._weather = WeatherState()
        self._opir = OPIRSatellite(cadence=opir_cadence)
        self._use_opir = use_opir
        self._max_thermal_dim = (
            max_thermal_dim if max_thermal_dim is not None else self.DEFAULT_THERMAL_DIM
        )
        self._base_ignition_rate = base_ignition_rate
        self._weather_volatility = weather_volatility
        self._n_drones = n_drones
        self._suppression_effectiveness = suppression_effectiveness

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
        thermal_dim = min(self._grid.rows * self._grid.cols, self._max_thermal_dim)
        weather_dim = len(self._weather_stations) * 5
        fuel_dim = len(self._fuel_sensors) * 3

        self._streams = [
            Stream(
                stream_type=StreamType.RAW,
                dimensionality=thermal_dim,
                label="thermal_detection",
                current_data=np.zeros(thermal_dim),
                metadata=self._thermal_metadata(thermal_dim),
            ),
            Stream(
                stream_type=StreamType.RAW,
                dimensionality=weather_dim,
                label="weather_observations",
                current_data=np.zeros(weather_dim),
                metadata=self._weather_metadata(),
            ),
            Stream(
                stream_type=StreamType.RAW,
                dimensionality=fuel_dim,
                label="fuel_moisture",
                current_data=np.zeros(fuel_dim),
                metadata=self._fuel_metadata(),
            ),
        ]

    def _thermal_metadata(self, dimensionality: int) -> StreamMetadata:
        """Declare sampled thermal cells and their coarse spatial support."""
        indices = self._thermal_sample_indices(dimensionality)
        coordinates: list[tuple[float, ...] | None] = [
            (float(index // self._grid.cols), float(index % self._grid.cols)) for index in indices
        ]
        stride = (
            float(np.sqrt(self._grid.rows * self._grid.cols / dimensionality))
            if dimensionality < self._grid.rows * self._grid.cols
            else 1.0
        )
        return StreamMetadata(
            coordinates=coordinates,
            sensor_coordinates=list(coordinates),
            modality=["thermal"] * dimensionality,
            identity=[None] * dimensionality,
            footprints=[(stride, stride)] * dimensionality,
            resolution=[stride] * dimensionality,
        )

    def _weather_metadata(self) -> StreamMetadata:
        """Declare localized weather features for each fixed station."""
        modalities = ("temperature", "humidity", "wind_speed", "wind_direction", "precipitation")
        sensor_coordinates: list[tuple[float, ...] | None] = [
            (float(station.row), float(station.col))
            for station in self._weather_stations
            for _ in modalities
        ]
        return StreamMetadata(
            coordinates=[None] * len(sensor_coordinates),
            sensor_coordinates=sensor_coordinates,
            modality=[modality for _ in self._weather_stations for modality in modalities],
            identity=[None] * len(sensor_coordinates),
            footprints=[(0.0, 0.0)] * len(sensor_coordinates),
            resolution=[0.0] * len(sensor_coordinates),
        )

    def _fuel_metadata(self) -> StreamMetadata:
        """Declare localized fuel-moisture features for each fixed probe."""
        modalities = ("live_moisture", "dead_moisture", "effective_moisture")
        sensor_coordinates: list[tuple[float, ...] | None] = [
            (float(sensor.row), float(sensor.col))
            for sensor in self._fuel_sensors
            for _ in modalities
        ]
        return StreamMetadata(
            coordinates=[None] * len(sensor_coordinates),
            sensor_coordinates=sensor_coordinates,
            modality=[modality for _ in self._fuel_sensors for modality in modalities],
            identity=[None] * len(sensor_coordinates),
            footprints=[(0.0, 0.0)] * len(sensor_coordinates),
            resolution=[0.0] * len(sensor_coordinates),
        )

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
        self._grid.stochastic_ignition(
            self._weather, time_step, self.rng, base_rate=self._base_ignition_rate
        )
        self._grid.step(self._weather, time_step, self.rng)
        self._update_fuel_moisture()
        self._update_streams(time_step, fire_grid=self._grid, rng=self.rng)

    def observe_grid(
        self,
        fire_grid: FireGrid,
        weather: WeatherState,
        time_step: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Populate sensor streams from an external grid without advancing physics.

        Use when a comparison harness or other caller owns the canonical FireGrid
        (e.g. A4 BMA inside ``run_comparison``). The adapter acts as a sensor
        front-end only; it does not step its internal grid.
        """
        self._current_step = time_step
        self._weather = weather
        self._update_streams(time_step, fire_grid=fire_grid, rng=rng or self.rng)

    def dispatch_drone_suppression(self, n_correct_reports: int) -> int:
        """Dispatch drones to suppress fires after verified agent detections.

        Legacy count-based API; prefer ``dispatch_and_judge_responses`` for
        location-targeted dispatch with post-dispatch judgment.
        """
        if self._n_drones <= 0 or n_correct_reports <= 0:
            return 0
        max_dispatches = min(self._n_drones, n_correct_reports)
        suppressed = 0
        for row, col in self._grid.active_fire_cells()[:max_dispatches]:
            if self._grid.suppress(row, col, self._suppression_effectiveness):
                suppressed += 1
        return suppressed

    def get_responder_user_id(self) -> str:
        """Fire Operations Chief authorizes suppression dispatch."""
        for user in self._users:
            if user.name == "Fire Operations Chief":
                return user.id
        return self._users[1].id if len(self._users) > 1 else self._users[0].id

    def dispatch_and_judge_responses(
        self,
        targets: list[DispatchTarget],
        time_step: int,
    ) -> list[ResponseOutcome]:
        """Suppress fires at COP-selected locations and judge responder necessity."""
        outcomes: list[ResponseOutcome] = []
        dispatches_remaining = self._n_drones
        responder_id = self.get_responder_user_id()

        for target in targets:
            row, col = target.location
            before = self._fire_severity(row, col)
            dispatched = False
            after = before

            if self._n_drones > 0 and dispatches_remaining > 0:
                self._grid.suppress(row, col, self._suppression_effectiveness)
                after = self._fire_severity(row, col)
                dispatched = True
                dispatches_remaining -= 1

            problem, mitigated, necessary = judge_necessity(before, after)
            linked_reports = target.reports or [
                Report(
                    agent_id="",
                    target_user_id=responder_id,
                    time_step=time_step,
                    signal_vector=np.array([]),
                    confidence=0.0,
                    anomaly_score=0.0,
                    location=target.location,
                    verified=True,
                )
            ]
            for report in linked_reports:
                outcome = ResponseOutcome(
                    agent_id=report.agent_id,
                    responder_user_id=responder_id,
                    time_step=time_step,
                    location=target.location,
                    response_type="suppression",
                    dispatched=dispatched,
                    problem_severity_before=before,
                    problem_severity_after=after,
                    problem_present=problem,
                    mitigated=mitigated,
                    response_necessary=necessary,
                )
                report.response_outcome = outcome
                outcomes.append(outcome)

        return outcomes

    def _fire_severity(self, row: int, col: int) -> float:
        """Fire intensity at a grid cell (0 if not burning)."""
        if row < 0 or col < 0 or row >= self._grid.rows or col >= self._grid.cols:
            return 0.0
        fs = self._grid.fire[row][col]
        return float(fs.intensity) if fs.is_active else 0.0

    def get_ground_truth(self, time_step: int) -> bool:
        """A fire event is active if any cells are currently burning."""
        return len(self._grid.active_fire_cells()) > 0

    def get_active_locations(self, time_step: int) -> list[EventLocation]:
        """Return grid coordinates of all currently burning cells."""
        return self._grid.active_fire_cells()

    def get_location_frame(self) -> tuple[EventLocation, EventLocation]:
        """Declare the inclusive FireGrid coordinate frame."""
        return ((0, 0), (self._grid.rows - 1, self._grid.cols - 1))

    def infer_report_location(
        self,
        stream_data: list[NDArray[np.float64]],
        stream_labels: list[str],
    ) -> EventLocation:
        """Infer fire location from the sampled thermal stream peak.

        Each thermal value represents one sampled grid cell; unsampled cells
        are outside this stream's spatial resolution.
        """
        for data, label in zip(stream_data, stream_labels, strict=False):
            if label == "thermal_detection" and data.size > 0:
                peak_idx = int(np.argmax(data))
                cols = self._grid.cols
                full_idx = int(self._thermal_sample_indices(data.size)[peak_idx])
                return (full_idx // cols, full_idx % cols)
        return (0, 0)

    def score_relevance(self, signal_vector: NDArray[np.float64], user: User) -> float:
        from tattletots.engine.relevance import score_report_relevance

        return score_report_relevance(signal_vector, user)

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
        """Sinusoidal weather with noise scaled by weather_volatility."""
        phase = 2.0 * np.pi * time_step / 200.0
        v = self._weather_volatility
        self._weather = WeatherState(
            temperature=25.0 + 10.0 * np.sin(phase) + float(self.rng.normal(0, 2 * v)),
            humidity=float(np.clip(0.4 + 0.2 * np.cos(phase) + self.rng.normal(0, 0.05 * v), 0, 1)),
            wind_speed=max(0.0, 5.0 + 3.0 * np.sin(phase * 0.7) + float(self.rng.normal(0, 1 * v))),
            wind_direction=float((180.0 + 90.0 * np.sin(phase * 0.3)) % 360),
            precipitation=max(
                0.0,
                float(self.rng.exponential(0.5 * v) if self.rng.random() < 0.1 else 0.0),
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

    def _update_streams(
        self,
        time_step: int,
        *,
        fire_grid: FireGrid | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Populate stream data from sensor outputs."""
        grid = fire_grid if fire_grid is not None else self._grid
        observe_rng = rng if rng is not None else self.rng

        thermal = self._build_thermal_vector(time_step, grid, observe_rng)
        thermal_status = self._thermal_status(time_step, grid)
        thermal_stream = self._streams[0]
        if thermal_stream.metadata is not None:
            thermal_coordinates = self._thermal_metadata(thermal_stream.dimensionality).coordinates
            if thermal_coordinates is None:
                raise RuntimeError("thermal metadata must declare sampled coordinates")
            thermal_stream.metadata = thermal_stream.metadata.model_copy(
                update={
                    "coordinates": [
                        coordinate if status == ObservationStatus.OBSERVED.value else None
                        for coordinate, status in zip(
                            thermal_coordinates,
                            thermal_status,
                            strict=True,
                        )
                    ]
                }
            )
        thermal_stream.update(thermal, thermal_status)

        weather_obs = np.concatenate(
            [ws.observe(self._weather, observe_rng) for ws in self._weather_stations]
        )
        self._streams[1].update(
            weather_obs,
            np.full(
                weather_obs.size,
                ObservationStatus.OBSERVED.value,
                dtype="<U8",
            ),
        )

        fuel_parts: list[np.ndarray] = []
        for fs in self._fuel_sensors:
            obs = fs.observe(grid.fuel[fs.row][fs.col], time_step, observe_rng)
            fuel_parts.append(obs if obs is not None else np.zeros(3))
        fuel_data = np.concatenate(fuel_parts)
        fuel_status = np.concatenate(
            [
                np.full(
                    3,
                    (
                        ObservationStatus.OBSERVED.value
                        if time_step % sensor.cadence == 0
                        else ObservationStatus.MISSING.value
                    ),
                    dtype="<U8",
                )
                for sensor in self._fuel_sensors
            ]
        )
        self._streams[2].update(fuel_data, fuel_status)

    def _thermal_status(self, time_step: int, grid: FireGrid) -> np.ndarray:
        """Declare thermal coverage from cadence, placement, range, and LOS."""
        indices = self._thermal_sample_indices(self._streams[0].dimensionality)
        statuses: list[str] = []
        opir_available = self._use_opir and time_step % self._opir.cadence == 0
        for index in indices:
            row, col = divmod(int(index), self._grid.cols)
            camera_available = any(
                camera.covers_cell(row, col, grid.terrain) for camera in self._cameras
            )
            statuses.append(
                ObservationStatus.OBSERVED.value
                if opir_available or camera_available
                else ObservationStatus.MISSING.value
            )
        return np.asarray(statuses, dtype="<U8")

    def _build_thermal_vector(
        self,
        time_step: int,
        fire_grid: FireGrid | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Fuse OPIR and camera detections into the sampled thermal stream."""
        grid = fire_grid if fire_grid is not None else self._grid
        observe_rng = rng if rng is not None else self.rng
        dim = self._streams[0].dimensionality
        detections: dict[tuple[int, int], float] = {}
        if self._use_opir:
            sensor_detections = self._opir.scan(grid, time_step, observe_rng)
            for row, col, confidence in sensor_detections:
                detections[(row, col)] = max(detections.get((row, col), 0.0), confidence)

        is_night = is_night_time(time_step)
        for camera in self._cameras:
            sensor_detections = camera.detect(grid, is_night, observe_rng)
            for row, col, confidence in sensor_detections:
                detections[(row, col)] = max(detections.get((row, col), 0.0), confidence)

        return np.asarray(
            [
                detections.get((int(index) // grid.cols, int(index) % grid.cols), 0.0)
                for index in self._thermal_sample_indices(dim)
            ],
            dtype=float,
        )

    def _thermal_sample_indices(self, thermal_dim: int | None = None) -> NDArray[np.int64]:
        """Return full-grid indices represented by thermal stream positions."""
        total_cells = self._grid.rows * self._grid.cols
        dim = self._streams[0].dimensionality if thermal_dim is None else thermal_dim
        if total_cells <= dim:
            return np.arange(total_cells, dtype=np.int64)
        return np.linspace(0, total_cells - 1, dim, dtype=int)

    @property
    def n_drones(self) -> int:
        """Operational drone fleet size for post-detection suppression."""
        return self._n_drones

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
