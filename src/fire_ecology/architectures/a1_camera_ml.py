"""A1: Camera/ML network — ALERT-style camera network + smoke ML + human verification."""

from __future__ import annotations

import numpy as np

from fire_ecology.architectures.base import Architecture, ArchitectureResult
from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.camera_tower import CameraTower
from fire_ecology.sensors.opir import OPIRSatellite


class CameraMLNetwork(Architecture):
    """Camera network with ML-based smoke/fire detection.

    Fixed camera towers detect fires via computer vision, with human
    verification before dispatch. OPIR provides satellite backstop.
    """

    def __init__(
        self,
        cameras: list[CameraTower] | None = None,
        ml_confidence_threshold: float = 0.6,
        human_verification_delay: int = 1,
    ) -> None:
        self.cameras = cameras or []
        self.ml_confidence_threshold = ml_confidence_threshold
        self.human_verification_delay = human_verification_delay
        self._pending: list[tuple[int, int, int]] = []

    def step(
        self,
        fire_grid: FireGrid,
        weather: WeatherState,
        opir: OPIRSatellite,
        time_step: int,
        rng: np.random.Generator,
    ) -> ArchitectureResult:
        detections: list[tuple[int, int]] = []
        suppressions: list[tuple[int, int]] = []
        cost = len(self.cameras) * 0.1

        is_night = (time_step % 24) >= 18 or (time_step % 24) < 6
        for camera in self.cameras:
            hits = camera.detect(fire_grid, is_night, rng)
            for r, c, conf in hits:
                if conf >= self.ml_confidence_threshold and (r, c) not in detections:
                    detections.append((r, c))
                    self._pending.append((r, c, time_step))

        opir_hits = opir.scan(fire_grid, time_step, rng)
        for r, c, _conf in opir_hits:
            if (r, c) not in detections:
                detections.append((r, c))
                self._pending.append((r, c, time_step))

        ready = [
            (r, c) for r, c, t in self._pending if time_step - t >= self.human_verification_delay
        ]
        for r, c in ready:
            if fire_grid.suppress(r, c, effectiveness=0.6):
                suppressions.append((r, c))
                cost += 3.0

        self._pending = [
            (r, c, t)
            for r, c, t in self._pending
            if time_step - t < self.human_verification_delay + 5
        ]

        return ArchitectureResult(
            detections=detections,
            suppressions=suppressions,
            escalations=len(detections),
            cost=cost,
        )

    def reset(self) -> None:
        self._pending = []
