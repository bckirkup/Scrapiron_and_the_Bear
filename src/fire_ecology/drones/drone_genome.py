"""Drone genome: heritable behavioral traits layered on top of body plan hardware."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, Field

from fire_ecology.drones.body_plan import BodyPlan, BodyPlanType


class DroneGenome(BaseModel):
    """Heritable traits for a fire-ecology drone Tot.

    These behavioral parameters evolve within the constraints set by the
    drone's physical body plan. The body plan itself does not mutate.
    """

    model_config = {"arbitrary_types_allowed": True}

    body_plan: BodyPlan = Field(default_factory=BodyPlan)

    patrol_fraction: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Fraction of energy budget allocated to patrol/scouting",
    )
    suppress_fraction: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of energy budget allocated to suppression",
    )
    report_fraction: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of energy budget allocated to reporting",
    )
    preferred_terrain_risk: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Preferred risk level for patrol niche (0=low risk, 1=hotspots)",
    )
    escalation_to_drone: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Anomaly threshold to escalate to other drones",
    )
    escalation_to_human: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Anomaly threshold to escalate to human operators",
    )
    swap_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Battery fraction at which drone returns for SWAP",
    )
    water_reload_eagerness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How eagerly the drone seeks water reload (0=lazy, 1=eager)",
    )

    @property
    def expected_role(self) -> str:
        """Infer the emergent role from genome and body plan."""
        if self.body_plan.plan_type == BodyPlanType.RELAY:
            return "relay"
        if self.body_plan.tank_gallons >= 20.0 and self.suppress_fraction > 0.4:
            return "strike"
        if self.report_fraction > 0.5:
            return "broker"
        if self.body_plan.sensor_count >= 3 and self.patrol_fraction > 0.4:
            return "scout"
        return "generalist"

    def mutate(self, rng: np.random.Generator, rate: float = 0.1) -> Self:
        """Return a mutated copy. Body plan is NOT mutated (hardware is fixed)."""
        data = self.model_dump()
        float_traits = [
            "patrol_fraction",
            "suppress_fraction",
            "report_fraction",
            "preferred_terrain_risk",
            "escalation_to_drone",
            "escalation_to_human",
            "swap_threshold",
            "water_reload_eagerness",
        ]
        for trait in float_traits:
            if rng.random() < rate:
                data[trait] = float(np.clip(data[trait] + rng.normal(0, 0.05), 0.0, 1.0))

        total = data["patrol_fraction"] + data["suppress_fraction"] + data["report_fraction"]
        if total > 0:
            data["patrol_fraction"] /= total
            data["suppress_fraction"] /= total
            data["report_fraction"] /= total

        return type(self).model_validate(data)
