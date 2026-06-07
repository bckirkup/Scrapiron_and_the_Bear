"""Terrain model: elevation, slope, aspect for each grid cell."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TerrainCell(BaseModel):
    """Static terrain properties for a single grid cell.

    Slope and aspect influence fire spread rate and direction.
    Elevation affects weather interpolation and line-of-sight calculations.
    """

    elevation: float = Field(default=0.0, description="Elevation in meters above sea level")
    slope: float = Field(default=0.0, ge=0.0, le=90.0, description="Slope in degrees")
    aspect: float = Field(
        default=0.0,
        ge=0.0,
        lt=360.0,
        description="Aspect in degrees from north (0=N, 90=E, 180=S, 270=W)",
    )
    is_water: bool = Field(default=False, description="Whether this cell is a water body")
    is_road: bool = Field(default=False, description="Whether this cell contains a road")
    is_firebreak: bool = Field(default=False, description="Whether this cell is a firebreak")
    is_structure: bool = Field(default=False, description="Whether this cell has infrastructure")

    @property
    def is_burnable(self) -> bool:
        """A cell can burn only if it is not water, road, or firebreak."""
        return not (self.is_water or self.is_road or self.is_firebreak)
