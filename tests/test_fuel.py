"""Tests for fuel model."""

from __future__ import annotations

import pytest

from fire_ecology.environment.fuel import FuelCell, FuelType


class TestFuelType:
    def test_grass_spread_rate(self) -> None:
        assert FuelType.GRASS.base_spread_rate == 1.0

    def test_barren_no_spread(self) -> None:
        assert FuelType.BARREN.base_spread_rate == 0.0

    def test_timber_highest_intensity(self) -> None:
        assert FuelType.TIMBER.base_intensity == 1.0

    def test_barren_no_intensity(self) -> None:
        assert FuelType.BARREN.base_intensity == 0.0

    def test_shrub_intermediate(self) -> None:
        assert 0.0 < FuelType.SHRUB.base_spread_rate < FuelType.GRASS.base_spread_rate
        assert FuelType.SHRUB.base_intensity > FuelType.GRASS.base_intensity


class TestFuelCell:
    def test_effective_moisture(self) -> None:
        cell = FuelCell(live_moisture=0.5, dead_moisture=0.3)
        expected = 0.4 * 0.5 + 0.6 * 0.3
        assert cell.effective_moisture == pytest.approx(expected)

    def test_spread_modifier_dry(self) -> None:
        cell = FuelCell(fuel_type=FuelType.GRASS, live_moisture=0.0, dead_moisture=0.0)
        assert cell.spread_modifier > 0.0

    def test_spread_modifier_wet(self) -> None:
        cell = FuelCell(fuel_type=FuelType.GRASS, live_moisture=1.0, dead_moisture=1.0)
        assert cell.spread_modifier == 0.0

    def test_intensity_modifier(self) -> None:
        cell = FuelCell(fuel_type=FuelType.TIMBER, fuel_load=0.8)
        assert cell.intensity_modifier == pytest.approx(0.8)

    def test_barren_no_spread(self) -> None:
        cell = FuelCell(fuel_type=FuelType.BARREN)
        assert cell.spread_modifier == 0.0
        assert cell.intensity_modifier == 0.0
