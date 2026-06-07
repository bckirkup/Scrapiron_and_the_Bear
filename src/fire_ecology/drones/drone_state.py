"""Drone runtime state: position, battery, water, sortie tracking."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DroneState(BaseModel):
    """Mutable runtime state for a drone Tot (not heritable)."""

    row: int = Field(default=0, description="Current grid row")
    col: int = Field(default=0, description="Current grid col")
    battery: float = Field(default=1.0, ge=0.0, le=1.0, description="Remaining battery fraction")
    water: float = Field(default=1.0, ge=0.0, le=1.0, description="Remaining water fraction")
    flight_steps: int = Field(default=0, ge=0, description="Steps flown since last SWAP")
    sorties_completed: int = Field(default=0, ge=0, description="Total completed sorties")
    suppression_attempts: int = Field(default=0, ge=0, description="Lifetime suppression attempts")
    successful_suppressions: int = Field(
        default=0, ge=0, description="Lifetime successful suppressions"
    )
    is_returning: bool = Field(
        default=False, description="Whether drone is returning to base for SWAP/reload"
    )

    @property
    def needs_swap(self) -> bool:
        """Whether battery is critically low."""
        return self.battery < 0.1

    @property
    def needs_water(self) -> bool:
        """Whether water tank is empty."""
        return self.water <= 0.0
