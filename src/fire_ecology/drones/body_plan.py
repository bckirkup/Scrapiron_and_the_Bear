"""Drone body plan: hardware template defining payload, sensor, and endurance."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class BodyPlanType(enum.StrEnum):
    """Drone body plan archetypes.

    SCOUT: small, high endurance, rich sensing, no water payload.
    STRIKE_SMALL: 5-gallon payload, moderate sensing.
    STRIKE_LARGE: 40-gallon payload, minimal sensing, high power cost.
    RELAY: communication relay, minimal sensing, high endurance.
    HYBRID: balanced scout/strike with moderate payload and sensing.
    """

    SCOUT = "scout"
    STRIKE_SMALL = "strike_small"
    STRIKE_LARGE = "strike_large"
    RELAY = "relay"
    HYBRID = "hybrid"


class BodyPlan(BaseModel):
    """Physical hardware template for a drone Tot.

    Defines the fixed hardware capabilities that constrain what a drone
    can do. The genome evolves behavioral parameters within these limits.
    """

    plan_type: BodyPlanType = Field(
        default=BodyPlanType.SCOUT,
        description="Hardware archetype",
    )
    tank_gallons: float = Field(
        default=0.0, ge=0.0, le=40.0, description="Water tank capacity in gallons"
    )
    battery_capacity: float = Field(
        default=1.0, gt=0.0, description="Battery capacity (normalized, 1.0 = standard)"
    )
    max_speed: float = Field(
        default=1.0, gt=0.0, description="Maximum speed in grid cells per step"
    )
    endurance: int = Field(
        default=50, ge=1, description="Maximum flight time in steps before SWAP required"
    )
    has_rgb: bool = Field(default=True, description="RGB camera")
    has_thermal: bool = Field(default=False, description="Thermal imaging sensor")
    has_multispectral: bool = Field(default=False, description="Multispectral sensor")
    has_smoke_pm: bool = Field(default=False, description="Smoke/PM particulate sensor")
    has_weather_probe: bool = Field(default=False, description="Weather microprobe")

    @property
    def sensor_count(self) -> int:
        """Number of sensors on this drone."""
        return sum(
            [
                self.has_rgb,
                self.has_thermal,
                self.has_multispectral,
                self.has_smoke_pm,
                self.has_weather_probe,
            ]
        )

    @property
    def can_suppress(self) -> bool:
        """Whether this drone can perform fire suppression."""
        return self.tank_gallons > 0.0

    @property
    def suppression_effectiveness(self) -> float:
        """Effectiveness of this body plan's suppression capability.

        Scales with tank size: 5-gal → 0.4, 40-gal → 0.85.
        Returns 0.0 for drones without a water tank.
        """
        if self.tank_gallons <= 0.0:
            return 0.0
        return min(0.2 + 0.0175 * self.tank_gallons, 0.95)

    @classmethod
    def scout(cls) -> BodyPlan:
        """Factory for scout drone: high endurance, rich sensing, no water."""
        return cls(
            plan_type=BodyPlanType.SCOUT,
            tank_gallons=0.0,
            battery_capacity=1.2,
            max_speed=1.5,
            endurance=80,
            has_rgb=True,
            has_thermal=True,
            has_multispectral=True,
            has_smoke_pm=True,
            has_weather_probe=True,
        )

    @classmethod
    def strike_small(cls) -> BodyPlan:
        """Factory for small strike drone: 5-gallon payload."""
        return cls(
            plan_type=BodyPlanType.STRIKE_SMALL,
            tank_gallons=5.0,
            battery_capacity=1.0,
            max_speed=1.0,
            endurance=40,
            has_rgb=True,
            has_thermal=True,
        )

    @classmethod
    def strike_large(cls) -> BodyPlan:
        """Factory for large strike drone: 40-gallon payload."""
        return cls(
            plan_type=BodyPlanType.STRIKE_LARGE,
            tank_gallons=40.0,
            battery_capacity=1.5,
            max_speed=0.7,
            endurance=25,
            has_rgb=True,
            has_thermal=True,
        )

    @classmethod
    def relay(cls) -> BodyPlan:
        """Factory for relay drone: communications, high endurance."""
        return cls(
            plan_type=BodyPlanType.RELAY,
            tank_gallons=0.0,
            battery_capacity=1.3,
            max_speed=0.8,
            endurance=100,
            has_rgb=True,
        )

    @classmethod
    def hybrid(cls) -> BodyPlan:
        """Factory for hybrid drone: moderate everything."""
        return cls(
            plan_type=BodyPlanType.HYBRID,
            tank_gallons=5.0,
            battery_capacity=1.1,
            max_speed=1.2,
            endurance=50,
            has_rgb=True,
            has_thermal=True,
            has_smoke_pm=True,
        )
