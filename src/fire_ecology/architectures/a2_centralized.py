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
        weather: WeatherState,
        opir: OPIRSatellite,
        time_step: int,
        rng: np.random.Generator,
    ) -> ArchitectureResult:
        if not self._drone_positions:
            self._initialize_positions(fire_grid, rng)

        detections: list[tuple[int, int]] = []
        suppressions: list[tuple[int, int]] = []
        cost = self.n_drones * 0.5

        for dr, dc in self._drone_positions:
            for r in range(
                max(0, dr - self.detection_range),
                min(fire_grid.rows, dr + self.detection_range + 1),
            ):
                for c in range(
                    max(0, dc - self.detection_range),
                    min(fire_grid.cols, dc + self.detection_range + 1),
                ):
                    if fire_grid.fire[r][c].is_active and (r, c) not in detections:
                        detections.append((r, c))

        opir_hits = opir.scan(fire_grid, time_step, rng)
        for r, c, _conf in opir_hits:
            if (r, c) not in detections:
                detections.append((r, c))

        if detections:
            self._assign_drones_to_fires(detections, fire_grid, rng)

        for r, c in detections[: self.n_drones]:
            if fire_grid.suppress(r, c, effectiveness=self.suppression_effectiveness):
                suppressions.append((r, c))
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
        rng: np.random.Generator,
    ) -> None:
        """Move drones toward detected fires (greedy nearest assignment)."""
        new_positions: list[tuple[int, int]] = []
        assigned_fires = set[int]()

        for i in range(len(self._drone_positions)):
            if not fire_cells:
                new_positions.append(self._drone_positions[i])
                continue
            best_dist = float("inf")
            best_idx = 0
            dr, dc = self._drone_positions[i]
            for j, (fr, fc) in enumerate(fire_cells):
                if j in assigned_fires:
                    continue
                d = abs(dr - fr) + abs(dc - fc)
                if d < best_dist:
                    best_dist = d
                    best_idx = j
            assigned_fires.add(best_idx)
            new_positions.append(fire_cells[best_idx])

        self._drone_positions = new_positions

    def reset(self) -> None:
        self._drone_positions = []
