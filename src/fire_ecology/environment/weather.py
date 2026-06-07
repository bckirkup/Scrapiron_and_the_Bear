"""Weather model: temperature, humidity, wind, precipitation per time step."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class WeatherState(BaseModel):
    """Weather conditions for the entire grid at a single time step.

    Wind direction/speed affect fire spread direction and rate.
    Humidity and precipitation affect fuel moisture dynamics.
    Temperature influences fire weather index.
    """

    model_config = {"arbitrary_types_allowed": True}

    temperature: float = Field(default=25.0, description="Air temperature in Celsius")
    humidity: float = Field(default=0.4, ge=0.0, le=1.0, description="Relative humidity (fraction)")
    wind_speed: float = Field(default=5.0, ge=0.0, description="Wind speed in m/s")
    wind_direction: float = Field(
        default=0.0,
        ge=0.0,
        lt=360.0,
        description="Wind direction in degrees from north",
    )
    precipitation: float = Field(default=0.0, ge=0.0, description="Precipitation in mm/step")

    @property
    def fire_weather_index(self) -> float:
        """Simplified fire weather index combining temperature, humidity, and wind.

        Higher values indicate greater fire danger. Range approximately 0-10.
        """
        temp_factor = max(0.0, (self.temperature - 10.0) / 30.0)
        dry_factor = 1.0 - self.humidity
        wind_factor = min(self.wind_speed / 20.0, 1.0)
        precip_suppression = 1.0 / (1.0 + self.precipitation)
        return 10.0 * temp_factor * dry_factor * wind_factor * precip_suppression

    @property
    def wind_vector(self) -> np.ndarray:
        """Wind as (east, north) components in m/s."""
        rad = np.radians(self.wind_direction)
        return np.array(
            [
                self.wind_speed * np.sin(rad),
                self.wind_speed * np.cos(rad),
            ]
        )

    def moisture_drying_rate(self) -> float:
        """Rate at which fuel moisture decreases per step (0-1)."""
        temp_effect = max(0.0, (self.temperature - 15.0) / 40.0)
        humidity_effect = 1.0 - self.humidity
        wind_effect = min(self.wind_speed / 15.0, 0.5)
        return min(0.05 * (temp_effect + humidity_effect + wind_effect), 0.15)
