"""Fire-domain metrics aligned with spec §9 falsification criteria."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fire_ecology.environment.fire import FireGrid


class StepMetrics(BaseModel):
    """Per-step metrics snapshot."""

    time_step: int = Field(ge=0)
    active_fires: int = Field(default=0, ge=0)
    burned_area: int = Field(default=0, ge=0)
    detections: int = Field(default=0, ge=0)
    false_positives: int = Field(default=0, ge=0)
    suppressions: int = Field(default=0, ge=0)
    escalations: int = Field(default=0, ge=0)
    opir_rescues: int = Field(default=0, ge=0)
    surveillance_cost: float = Field(default=0.0, ge=0.0)
    response_cost: float = Field(default=0.0, ge=0.0)
    damage_cost: float = Field(default=0.0, ge=0.0)


class FireMetrics(BaseModel):
    """Cumulative fire-domain metrics tracker.

    Tracks all spec §9 metrics:
    - Detection latency
    - Hotspot lead time
    - Dry-spot prediction precision/recall
    - Controlled burn escape detection latency
    - Autonomous suppression success rate
    - Human crew escalation rate
    - False dispatch rate
    - Drone sortie cost
    - OPIR rescue rate
    - Robustness to drone/sensor failure
    - Colonization curve
    """

    history: list[StepMetrics] = Field(default_factory=list)
    total_fires_started: int = Field(default=0, ge=0)
    total_fires_detected: int = Field(default=0, ge=0)
    total_suppressions_attempted: int = Field(default=0, ge=0)
    total_suppressions_successful: int = Field(default=0, ge=0)
    total_false_dispatches: int = Field(default=0, ge=0)
    total_opir_rescues: int = Field(default=0, ge=0)
    total_escalations: int = Field(default=0, ge=0)
    detection_latencies: list[int] = Field(default_factory=list)
    _detected_cells: set[tuple[int, int]] = set()

    model_config = {"arbitrary_types_allowed": True}

    def record_step(self, metrics: StepMetrics) -> None:
        """Record a single step's metrics."""
        self.history.append(metrics)
        self.total_fires_detected += metrics.detections
        self.total_suppressions_successful += metrics.suppressions
        self.total_false_dispatches += metrics.false_positives
        self.total_opir_rescues += metrics.opir_rescues
        self.total_escalations += metrics.escalations

    def record_detection_latency(self, latency: int) -> None:
        """Record the detection latency for a single fire event."""
        self.detection_latencies.append(latency)

    def record_detections_from_grid(
        self, detections: list[tuple[int, int]], fire_grid: FireGrid, time_step: int
    ) -> None:
        """Record detection latency for newly-detected fire cells.

        Uses the fire cell's ``ignition_step`` to compute how many steps
        elapsed between ignition and first detection.  Each cell is only
        counted once (tracked via ``_detected_cells``).
        """
        for r, c in detections:
            if (r, c) in self._detected_cells:
                continue
            fs = fire_grid.fire[r][c]
            if fs.ignition_step >= 0:
                self._detected_cells.add((r, c))
                self.record_detection_latency(time_step - fs.ignition_step)

    @property
    def mean_detection_latency(self) -> float:
        """Mean detection latency across all fire events."""
        if not self.detection_latencies:
            return float("inf")
        return sum(self.detection_latencies) / len(self.detection_latencies)

    @property
    def suppression_success_rate(self) -> float:
        """Fraction of suppression attempts that succeeded."""
        if self.total_suppressions_attempted == 0:
            return 0.0
        return self.total_suppressions_successful / self.total_suppressions_attempted

    @property
    def false_dispatch_rate(self) -> float:
        """Fraction of dispatches that were false alarms."""
        total = self.total_fires_detected + self.total_false_dispatches
        if total == 0:
            return 0.0
        return self.total_false_dispatches / total

    @property
    def opir_rescue_rate(self) -> float:
        """Fraction of fires first caught by OPIR after local systems missed."""
        if self.total_fires_detected == 0:
            return 0.0
        return self.total_opir_rescues / self.total_fires_detected

    @property
    def total_cost(self) -> float:
        """Sum of all cost categories across all steps."""
        return sum(m.surveillance_cost + m.response_cost + m.damage_cost for m in self.history)

    def summary(self) -> dict[str, float]:
        """Return a summary dictionary of key metrics."""
        return {
            "mean_detection_latency": self.mean_detection_latency,
            "suppression_success_rate": self.suppression_success_rate,
            "false_dispatch_rate": self.false_dispatch_rate,
            "opir_rescue_rate": self.opir_rescue_rate,
            "total_cost": self.total_cost,
            "total_fires_detected": float(self.total_fires_detected),
            "total_escalations": float(self.total_escalations),
            "total_burned_area": float(self.history[-1].burned_area if self.history else 0),
        }
