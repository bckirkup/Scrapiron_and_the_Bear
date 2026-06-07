"""Fuel moisture sensor: slow-cadence ground probe for dry-spot prediction."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from fire_ecology.environment.fuel import FuelCell


class FuelMoistureSensor(BaseModel):
    """Ground-based fuel moisture probe.

    Slow-updating sensor with high value for predicting dry spots before
    they become fire hazards. Typically deployed in high-risk areas.
    """

    row: int = Field(description="Grid row position")
    col: int = Field(description="Grid col position")
    cadence: int = Field(
        default=10,
        ge=1,
        description="Measurement interval in time steps (slow sensor)",
    )
    noise_std: float = Field(
        default=0.03,
        ge=0.0,
        description="Measurement noise standard deviation",
    )

    def observe(
        self,
        fuel: FuelCell,
        time_step: int,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Return a moisture observation, or None if off-cadence.

        Returns [live_moisture, dead_moisture, effective_moisture].
        """
        if time_step % self.cadence != 0:
            return None
        noise = rng.normal(0.0, self.noise_std, size=3)
        obs = (
            np.array(
                [
                    fuel.live_moisture,
                    fuel.dead_moisture,
                    fuel.effective_moisture,
                ]
            )
            + noise
        )
        return np.clip(obs, 0.0, 1.0)
