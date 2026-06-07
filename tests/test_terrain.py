"""Tests for terrain model."""

from __future__ import annotations

from fire_ecology.environment.terrain import TerrainCell


class TestTerrainCell:
    def test_default_is_burnable(self) -> None:
        cell = TerrainCell()
        assert cell.is_burnable

    def test_water_not_burnable(self) -> None:
        cell = TerrainCell(is_water=True)
        assert not cell.is_burnable

    def test_road_not_burnable(self) -> None:
        cell = TerrainCell(is_road=True)
        assert not cell.is_burnable

    def test_firebreak_not_burnable(self) -> None:
        cell = TerrainCell(is_firebreak=True)
        assert not cell.is_burnable

    def test_structure_is_burnable(self) -> None:
        cell = TerrainCell(is_structure=True)
        assert cell.is_burnable

    def test_elevation_range(self) -> None:
        cell = TerrainCell(elevation=1500.0)
        assert cell.elevation == 1500.0

    def test_slope_range(self) -> None:
        cell = TerrainCell(slope=45.0)
        assert cell.slope == 45.0

    def test_aspect_range(self) -> None:
        cell = TerrainCell(aspect=270.0)
        assert cell.aspect == 270.0
