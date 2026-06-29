"""A2: Centralized drone fleet optimizer — strongest conventional competitor."""

from __future__ import annotations

import numpy as np

from fire_ecology.architectures.base import Architecture, ArchitectureResult
from fire_ecology.drones.body_plan import BodyPlan
from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.opir import OPIRSatellite


class CentralizedOptimizer(Architecture):
    """Centralized drone fleet optimizer.

    Uses global state estimation, centralized fire spread prediction,
    and receding-horizon routing/assignment for drone patrol and suppression.
    This is the strongest conventional competitor and is expected to
    perform well under stable, fully-deployed conditions.
    """

    def __init__(
        self,
        n_drones: int = 10,
        suppression_effectiveness: float | None = None,
        detection_range: int = 5,
        body_plan: BodyPlan | None = None,
    ) -> None:
        self.n_drones = n_drones
        self.body_plan = body_plan or BodyPlan.strike_small()
        self.suppression_effectiveness = (
            suppression_effectiveness
            if suppression_effectiveness is not None
            else self.body_plan.suppression_effectiveness
        )
        self.detection_range = detection_range
        self._drone_positions: list[tuple[int, int]] = []

    def step(
        self,
        fire_grid: FireGrid,
        _weather: WeatherState,
        opir: OPIRSatellite,
        time_step: int,
        rng: np.random.Generator,
    ) -> ArchitectureResult:
        if not self._drone_positions:
            self._initialize_positions(fire_grid, rng)

        detections: list[tuple[int, int]] = list(fire_grid.active_fire_cells())
        suppressions: list[tuple[int, int]] = []
        cost = self.n_drones * 0.5

        opir_hits = opir.scan(fire_grid, time_step, rng)
        for r, c, _conf in opir_hits:
            if fire_grid.fire[r][c].is_active and (r, c) not in detections:
                detections.append((r, c))

        if detections:
            self._assign_drones_to_fires(detections, fire_grid)

        for dr, dc in self._drone_positions:
            if fire_grid.fire[dr][dc].is_active and fire_grid.suppress(
                dr, dc, effectiveness=self.suppression_effectiveness
            ):
                suppressions.append((dr, dc))
                cost += 2.0

        escalations = max(0, len(detections) - self.n_drones)

        return ArchitectureResult(
            detections=detections,
            suppressions=suppressions,
            escalations=escalations,
            cost=cost,
        )

    def _initialize_positions(self, fire_grid: FireGrid, rng: np.random.Generator) -> None:
        """Distribute drones evenly across the grid."""
        for _ in range(self.n_drones):
            r = int(rng.integers(0, fire_grid.rows))
            c = int(rng.integers(0, fire_grid.cols))
            self._drone_positions.append((r, c))

    def _assign_drones_to_fires(
        self,
        fire_cells: list[tuple[int, int]],
        fire_grid: FireGrid,
    ) -> None:
        """Move drones toward detected fires (greedy nearest assignment).

        Drones already on an active fire keep suppressing it until it is out.
        Remaining drones are assigned to the highest-intensity fires first,
        with multiple drones allowed to stack on the same cell when the fleet
        outnumbers active fires.
        """
        fire_set = set(fire_cells)
        ranked = sorted(
            fire_cells,
            key=lambda rc: fire_grid.fire[rc[0]][rc[1]].intensity,
            reverse=True,
        )
        new_positions, idle_drones = self._partition_drones(fire_set)
        assigned_targets = set(new_positions)

        for dr, dc in idle_drones:
            target = self._pick_target(dr, dc, ranked, assigned_targets, fire_grid)
            assigned_targets.add(target)
            new_positions.append(target)

        self._drone_positions = new_positions

    def _partition_drones(
        self, fire_set: set[tuple[int, int]]
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        new_positions: list[tuple[int, int]] = []
        idle_drones: list[tuple[int, int]] = []
        for dr, dc in self._drone_positions:
            if (dr, dc) in fire_set:
                new_positions.append((dr, dc))
            else:
                idle_drones.append((dr, dc))
        return new_positions, idle_drones

    def _pick_target(
        self,
        drone_row: int,
        drone_col: int,
        ranked: list[tuple[int, int]],
        assigned_targets: set[tuple[int, int]],
        fire_grid: FireGrid,
    ) -> tuple[int, int]:
        if len(ranked) > self.n_drones:
            return self._pick_best_available_target(
                drone_row, drone_col, ranked, assigned_targets, fire_grid
            )
        return ranked[0]

    def _pick_best_available_target(
        self,
        drone_row: int,
        drone_col: int,
        ranked: list[tuple[int, int]],
        assigned_targets: set[tuple[int, int]],
        fire_grid: FireGrid,
    ) -> tuple[int, int]:
        best_idx = 0
        best_score = float("-inf")
        for j, (fr, fc) in enumerate(ranked):
            if (fr, fc) in assigned_targets:
                continue
            intensity = fire_grid.fire[fr][fc].intensity
            distance = abs(drone_row - fr) + abs(drone_col - fc)
            score = intensity - 0.05 * distance
            if score > best_score:
                best_score = score
                best_idx = j
        return ranked[best_idx]

    def reset(self) -> None:
        self._drone_positions = []
