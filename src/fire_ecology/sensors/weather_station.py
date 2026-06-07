"""Weather station sensor: localized temperature, humidity, wind, precipitation."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from fire_ecology.environment.weather import WeatherState


class WeatherStation(BaseModel):
    """Fixed weather observation station.

    Provides localized readings with measurement noise. Multiple stations
    enable spatial interpolation of weather across the grid.
    """

    row: int = Field(description="Grid row position")
    col: int = Field(description="Grid col position")
    noise_std: float = Field(
        default=0.05,
        ge=0.0,
        description="Standard deviation of Gaussian measurement noise",
    )

    def observe(
        self,
        weather: WeatherState,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return a noisy observation vector: [temp, humidity, wind_speed, wind_dir, precip]."""
        base = np.array(
            [
                weather.temperature,
                weather.humidity,
                weather.wind_speed,
                weather.wind_direction,
                weather.precipitation,
            ]
        )
        noise = rng.normal(0.0, self.noise_std, size=5)
        noisy = base + noise * np.array([2.0, 0.05, 1.0, 10.0, 0.5])
        noisy[1] = np.clip(noisy[1], 0.0, 1.0)
        noisy[2] = max(0.0, noisy[2])
        noisy[3] = noisy[3] % 360.0
        noisy[4] = max(0.0, noisy[4])
        result: np.ndarray = noisy
        return result
