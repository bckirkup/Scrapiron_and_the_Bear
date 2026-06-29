"""A3: Federated/edge network — local detection nodes with limited fusion."""

from __future__ import annotations

import numpy as np

from fire_ecology.architectures.base import Architecture, ArchitectureResult
from fire_ecology.drones.body_plan import BodyPlan
from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.opir import OPIRSatellite


class FederatedEdge(Architecture):
    """Federated edge network for fire detection.

    Local nodes detect and escalate independently. Limited cross-region
    fusion. Strong where bandwidth or central connectivity is limited,
    but weaker for global optimization.
    """

    def __init__(
        self,
        n_nodes: int = 4,
        node_range: int = 8,
        detection_prob: float = 0.75,
        suppression_effectiveness: float | None = None,
        body_plan: BodyPlan | None = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.node_range = node_range
        self.detection_prob = detection_prob
        self.body_plan = body_plan or BodyPlan.strike_small()
        self.suppression_effectiveness = (
            suppression_effectiveness
            if suppression_effectiveness is not None
            else self.body_plan.suppression_effectiveness
        )
        self._node_positions: list[tuple[int, int]] = []

    def step(
        self,
        fire_grid: FireGrid,
        _weather: WeatherState,
        opir: OPIRSatellite,
        time_step: int,
        rng: np.random.Generator,
    ) -> ArchitectureResult:
        if not self._node_positions:
            self._initialize_nodes(fire_grid)

        detections = self._scan_node_detections(fire_grid, rng)
        detections = self._merge_opir_detections(detections, opir, fire_grid, time_step, rng)
        suppressions, suppression_cost = self._apply_suppressions(detections, fire_grid)
        cost = self.n_nodes * 0.3 + suppression_cost

        return ArchitectureResult(
            detections=detections,
            suppressions=suppressions,
            escalations=len(detections),
            cost=cost,
        )

    def _scan_node_detections(
        self, fire_grid: FireGrid, rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        detections: list[tuple[int, int]] = []
        for nr, nc in self._node_positions:
            for r in range(
                max(0, nr - self.node_range),
                min(fire_grid.rows, nr + self.node_range + 1),
            ):
                for c in range(
                    max(0, nc - self.node_range),
                    min(fire_grid.cols, nc + self.node_range + 1),
                ):
                    fs = fire_grid.fire[r][c]
                    if (
                        fs.is_active
                        and rng.random() < self.detection_prob
                        and (r, c) not in detections
                    ):
                        detections.append((r, c))
        return detections

    def _merge_opir_detections(
        self,
        detections: list[tuple[int, int]],
        opir: OPIRSatellite,
        fire_grid: FireGrid,
        time_step: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        merged = list(detections)
        opir_hits = opir.scan(fire_grid, time_step, rng)
        for r, c, _conf in opir_hits:
            if (r, c) not in merged:
                merged.append((r, c))
        return merged

    def _apply_suppressions(
        self,
        detections: list[tuple[int, int]],
        fire_grid: FireGrid,
    ) -> tuple[list[tuple[int, int]], float]:
        suppressions: list[tuple[int, int]] = []
        cost = 0.0
        for r, c in detections:
            if fire_grid.suppress(r, c, effectiveness=self.suppression_effectiveness):
                suppressions.append((r, c))
                cost += 2.5
        return suppressions, cost

    def _initialize_nodes(self, fire_grid: FireGrid) -> None:
        """Distribute nodes in a grid pattern."""
        rows_per = max(1, fire_grid.rows // max(1, int(self.n_nodes**0.5)))
        cols_per = max(1, fire_grid.cols // max(1, int(self.n_nodes**0.5)))
        for i in range(self.n_nodes):
            r = min(
                (i // max(1, int(self.n_nodes**0.5))) * rows_per + rows_per // 2, fire_grid.rows - 1
            )
            c = min(
                (i % max(1, int(self.n_nodes**0.5))) * cols_per + cols_per // 2, fire_grid.cols - 1
            )
            self._node_positions.append((r, c))

    def reset(self) -> None:
        self._node_positions = []
