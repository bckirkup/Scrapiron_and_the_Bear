"""Camera tower sensor: fixed-position smoke/fire detection with terrain occlusion."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.terrain import TerrainCell


def is_night_time(time_step: int) -> bool:
    """Return whether the Fire simulation's conventional night window applies."""
    return (time_step % 24) >= 18 or (time_step % 24) < 6


class CameraTower(BaseModel):
    """Fixed camera tower for smoke/fire visual detection.

    Line-of-sight coverage with terrain occlusion. Worse performance at
    night and in fog/cloud conditions.
    """

    row: int = Field(description="Grid row position of the tower")
    col: int = Field(description="Grid col position of the tower")
    tower_height: float = Field(default=30.0, ge=1.0, description="Tower height in meters")
    max_range: float = Field(
        default=15.0, ge=1.0, description="Maximum detection range in grid cells"
    )
    detection_probability: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Base probability of detecting a fire within range and LOS",
    )
    night_penalty: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Multiplicative penalty during night/fog conditions",
    )

    def detect(
        self,
        fire_grid: FireGrid,
        is_night: bool,
        rng: np.random.Generator,
    ) -> list[tuple[int, int, float]]:
        """Scan visible area for fires. Returns list of (row, col, confidence)."""
        detections: list[tuple[int, int, float]] = []
        base_prob = self.detection_probability
        if is_night:
            base_prob *= self.night_penalty

        for r, c in fire_grid.active_fire_cells():
            if not self.covers_cell(r, c, fire_grid.terrain):
                continue
            dist = np.sqrt((r - self.row) ** 2 + (c - self.col) ** 2)
            distance_factor = 1.0 - (dist / self.max_range) * 0.5
            prob = base_prob * distance_factor
            if rng.random() < prob:
                conf = min(1.0, 0.6 + 0.3 * fire_grid.fire[r][c].intensity)
                detections.append((r, c, conf))

        return detections

    def covers_cell(
        self,
        target_row: int,
        target_col: int,
        terrain: list[list[TerrainCell]],
    ) -> bool:
        """Return whether this camera covers a target by range and line of sight."""
        distance = np.hypot(target_row - self.row, target_col - self.col)
        return distance <= self.max_range and self._has_line_of_sight(
            target_row, target_col, terrain
        )

    def _has_line_of_sight(
        self,
        target_row: int,
        target_col: int,
        terrain: list[list[TerrainCell]],
    ) -> bool:
        """Simplified LOS check: compare elevation angles along the line."""
        dr = target_row - self.row
        dc = target_col - self.col
        steps = max(abs(dr), abs(dc))
        if steps == 0:
            return True

        tower_elev = terrain[self.row][self.col].elevation + self.tower_height
        target_elev = terrain[target_row][target_col].elevation

        for i in range(1, steps):
            frac = i / steps
            ir = int(self.row + dr * frac)
            ic = int(self.col + dc * frac)
            blocking_elev = terrain[ir][ic].elevation
            los_elev = tower_elev + (target_elev - tower_elev) * frac
            if blocking_elev > los_elev:
                return False
        return True
