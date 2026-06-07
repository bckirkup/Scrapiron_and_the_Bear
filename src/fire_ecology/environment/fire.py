"""Fire dynamics: ignition, spread, intensity, and suppression.

Implements a simplified Rothermel-inspired cellular automaton where spread
probability depends on wind, slope, fuel, and moisture.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from fire_ecology.environment.fuel import FuelCell
from fire_ecology.environment.terrain import TerrainCell
from fire_ecology.environment.weather import WeatherState


class CellFireState(enum.StrEnum):
    """Fire state for a single cell."""

    UNBURNED = "unburned"
    BURNING = "burning"
    SMOLDERING = "smoldering"
    BURNED_OUT = "burned_out"


@dataclass
class FireState:
    """Per-cell fire state tracking."""

    state: CellFireState = CellFireState.UNBURNED
    intensity: float = 0.0
    burn_duration: int = 0
    ignition_step: int = -1

    @property
    def is_active(self) -> bool:
        return self.state in (CellFireState.BURNING, CellFireState.SMOLDERING)


@dataclass
class FireGrid:
    """Grid-level fire dynamics engine.

    Manages ignition, spread, and suppression across a 2-D grid of cells.
    Each cell references terrain, fuel, and fire state arrays by (row, col).
    """

    rows: int
    cols: int
    terrain: list[list[TerrainCell]] = field(default_factory=list)
    fuel: list[list[FuelCell]] = field(default_factory=list)
    fire: list[list[FireState]] = field(default_factory=list)
    controlled_burn_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:
        if not self.terrain:
            self.terrain = [[TerrainCell() for _ in range(self.cols)] for _ in range(self.rows)]
        if not self.fuel:
            self.fuel = [[FuelCell() for _ in range(self.cols)] for _ in range(self.rows)]
        if not self.fire:
            self.fire = [[FireState() for _ in range(self.cols)] for _ in range(self.rows)]
        if self.controlled_burn_mask.size == 0:
            self.controlled_burn_mask = np.zeros((self.rows, self.cols), dtype=bool)

    def ignite(self, row: int, col: int, time_step: int) -> bool:
        """Attempt to ignite a cell. Returns True if successful."""
        if not self._in_bounds(row, col):
            return False
        t = self.terrain[row][col]
        f = self.fire[row][col]
        if not t.is_burnable or f.state != CellFireState.UNBURNED:
            return False
        f.state = CellFireState.BURNING
        f.intensity = self.fuel[row][col].intensity_modifier
        f.ignition_step = time_step
        return True

    def step(self, weather: WeatherState, time_step: int, rng: np.random.Generator) -> int:
        """Advance fire dynamics by one step. Returns count of newly ignited cells."""
        newly_ignited = 0
        spread_candidates: list[tuple[int, int]] = []

        for r in range(self.rows):
            for c in range(self.cols):
                fs = self.fire[r][c]
                if fs.state == CellFireState.BURNING:
                    fs.burn_duration += 1
                    fuel_remaining = self.fuel[r][c].fuel_load
                    if fuel_remaining <= 0.01 or fs.burn_duration > 10:
                        fs.state = CellFireState.SMOLDERING
                    else:
                        self.fuel[r][c].fuel_load = max(0.0, fuel_remaining - 0.1 * fs.intensity)
                        for dr, dc in _NEIGHBORS:
                            nr, nc = r + dr, c + dc
                            if self._in_bounds(nr, nc):
                                spread_candidates.append((nr, nc))
                elif fs.state == CellFireState.SMOLDERING:
                    fs.burn_duration += 1
                    if fs.burn_duration > 15:
                        fs.state = CellFireState.BURNED_OUT

        for nr, nc in spread_candidates:
            if self.fire[nr][nc].state != CellFireState.UNBURNED:
                continue
            if not self.terrain[nr][nc].is_burnable:
                continue
            prob = self._spread_probability(nr, nc, weather)
            if rng.random() < prob:
                self.ignite(nr, nc, time_step)
                newly_ignited += 1

        return newly_ignited

    def suppress(self, row: int, col: int, effectiveness: float = 0.8) -> bool:
        """Apply suppression to a burning cell. Returns True if fire extinguished."""
        if not self._in_bounds(row, col):
            return False
        fs = self.fire[row][col]
        if not fs.is_active:
            return False
        fs.intensity *= 1.0 - effectiveness
        if fs.intensity < 0.05:
            fs.state = CellFireState.BURNED_OUT
            return True
        return False

    def stochastic_ignition(
        self, weather: WeatherState, time_step: int, rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        """Generate random ignitions from lightning or human causes."""
        base_rate = 0.0001
        fwi_factor = weather.fire_weather_index / 10.0
        ignition_rate = base_rate * (1.0 + fwi_factor)

        ignited: list[tuple[int, int]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (
                    self.fire[r][c].state == CellFireState.UNBURNED
                    and self.terrain[r][c].is_burnable
                    and rng.random() < ignition_rate
                    and self.ignite(r, c, time_step)
                ):
                    ignited.append((r, c))
        return ignited

    def active_fire_cells(self) -> list[tuple[int, int]]:
        """Return coordinates of all currently burning or smoldering cells."""
        return [
            (r, c) for r in range(self.rows) for c in range(self.cols) if self.fire[r][c].is_active
        ]

    def burned_area(self) -> int:
        """Total cells that have burned or are burning."""
        return sum(
            1
            for r in range(self.rows)
            for c in range(self.cols)
            if self.fire[r][c].state != CellFireState.UNBURNED
        )

    def _spread_probability(self, row: int, col: int, weather: WeatherState) -> float:
        """Compute spread probability for a target cell."""
        fuel = self.fuel[row][col]
        terrain = self.terrain[row][col]

        fuel_factor = fuel.spread_modifier
        slope_factor = 1.0 + 0.02 * terrain.slope
        wind_factor = 1.0 + 0.05 * weather.wind_speed

        base_prob = 0.15 * fuel_factor * slope_factor * wind_factor
        return min(base_prob, 0.95)

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols


_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
