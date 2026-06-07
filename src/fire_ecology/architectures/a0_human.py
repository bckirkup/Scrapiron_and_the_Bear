"""A0: Human/manual baseline — lookouts, patrols, conventional dispatch."""

from __future__ import annotations

import numpy as np

from fire_ecology.architectures.base import Architecture, ArchitectureResult
from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.opir import OPIRSatellite


class HumanBaseline(Architecture):
    """Human-only fire management.

    Relies on human lookouts, scheduled patrols, public reports, and
    conventional dispatch. OPIR provides satellite backstop.
    """

    def __init__(
        self,
        patrol_coverage: float = 0.1,
        detection_prob: float = 0.6,
        response_delay: int = 3,
    ) -> None:
        self.patrol_coverage = patrol_coverage
        self.detection_prob = detection_prob
        self.response_delay = response_delay
        self._pending_detections: list[tuple[int, int, int]] = []

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
        cost = 1.0

        active = fire_grid.active_fire_cells()
        n_patrol = max(1, int(len(active) * self.patrol_coverage)) if active else 0
        for r, c in active[:n_patrol]:
            if rng.random() < self.detection_prob:
                detections.append((r, c))
                self._pending_detections.append((r, c, time_step))

        opir_hits = opir.scan(fire_grid, time_step, rng)
        for r, c, _conf in opir_hits:
            if (r, c) not in detections:
                detections.append((r, c))
                self._pending_detections.append((r, c, time_step))

        ready = [
            (r, c) for r, c, t in self._pending_detections if time_step - t >= self.response_delay
        ]
        for r, c in ready:
            if fire_grid.suppress(r, c, effectiveness=0.5):
                suppressions.append((r, c))
                cost += 5.0

        self._pending_detections = [
            (r, c, t)
            for r, c, t in self._pending_detections
            if time_step - t < self.response_delay + 5
        ]

        escalations = len(detections)
        return ArchitectureResult(
            detections=detections,
            suppressions=suppressions,
            escalations=escalations,
            cost=cost,
        )

    def reset(self) -> None:
        self._pending_detections = []
