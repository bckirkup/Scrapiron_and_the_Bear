"""Fuel model: type, load, moisture for each grid cell."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class FuelType(enum.StrEnum):
    """Fuel classification following simplified Anderson fuel models."""

    GRASS = "grass"
    SHRUB = "shrub"
    TIMBER = "timber"
    BARREN = "barren"

    @property
    def base_spread_rate(self) -> float:
        """Relative spread rate multiplier (dimensionless)."""
        return _SPREAD_RATES[self]

    @property
    def base_intensity(self) -> float:
        """Relative fire intensity multiplier (dimensionless)."""
        return _INTENSITY[self]


_SPREAD_RATES: dict[FuelType, float] = {
    FuelType.GRASS: 1.0,
    FuelType.SHRUB: 0.6,
    FuelType.TIMBER: 0.3,
    FuelType.BARREN: 0.0,
}

_INTENSITY: dict[FuelType, float] = {
    FuelType.GRASS: 0.4,
    FuelType.SHRUB: 0.7,
    FuelType.TIMBER: 1.0,
    FuelType.BARREN: 0.0,
}


class FuelCell(BaseModel):
    """Dynamic fuel state for a single grid cell."""

    fuel_type: FuelType = Field(default=FuelType.GRASS, description="Fuel classification")
    fuel_load: float = Field(
        default=1.0, ge=0.0, description="Fuel load (tons/hectare, normalized 0-1)"
    )
    live_moisture: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Live fuel moisture content (fraction of dry weight)",
    )
    dead_moisture: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Dead fuel moisture content (fraction of dry weight)",
    )

    @property
    def effective_moisture(self) -> float:
        """Weighted moisture combining live and dead fuel."""
        return 0.4 * self.live_moisture + 0.6 * self.dead_moisture

    @property
    def spread_modifier(self) -> float:
        """Fuel-based spread rate modifier combining type, load, and moisture."""
        moisture_factor = max(0.0, 1.0 - self.effective_moisture)
        return self.fuel_type.base_spread_rate * self.fuel_load * moisture_factor

    @property
    def intensity_modifier(self) -> float:
        """Fuel-based intensity modifier."""
        return self.fuel_type.base_intensity * self.fuel_load
