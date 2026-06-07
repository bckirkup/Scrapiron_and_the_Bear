"""OPIR / satellite thermal sensor: always-available backstop for fire detection."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from fire_ecology.environment.fire import CellFireState, FireGrid


class OPIRSatellite(BaseModel):
    """Overhead Persistent Infrared satellite sensor.

    Provides periodic thermal scans of the entire grid. Detects active fires
    above a thermal threshold, with configurable false-positive and miss rates.
    Available to ALL architectures as a parallel backstop.
    """

    cadence: int = Field(default=5, ge=1, description="Scan interval in time steps (1, 5, or 15)")
    false_positive_rate: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Probability of detecting fire where none exists",
    )
    miss_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of failing to detect an active fire",
    )
    min_intensity_threshold: float = Field(
        default=0.2,
        ge=0.0,
        description="Minimum fire intensity for OPIR detection",
    )

    def scan(
        self,
        fire_grid: FireGrid,
        time_step: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int, float]]:
        """Perform a satellite scan. Returns list of (row, col, confidence).

        Only scans at cadence intervals. Returns empty list on off-cadence steps.
        """
        if time_step % self.cadence != 0:
            return []

        detections: list[tuple[int, int, float]] = []
        for r in range(fire_grid.rows):
            for c in range(fire_grid.cols):
                fs = fire_grid.fire[r][c]
                if fs.is_active and fs.intensity >= self.min_intensity_threshold:
                    if rng.random() > self.miss_rate:
                        conf = min(1.0, 0.5 + 0.5 * fs.intensity)
                        detections.append((r, c, conf))
                elif fs.state == CellFireState.UNBURNED and rng.random() < self.false_positive_rate:
                    detections.append((r, c, 0.3))

        return detections
