"""Tests for weather model."""

from __future__ import annotations

import pytest

from fire_ecology.environment.weather import WeatherState


class TestWeatherState:
    def test_fire_weather_index_calm(self) -> None:
        w = WeatherState(temperature=15.0, humidity=0.8, wind_speed=2.0)
        assert w.fire_weather_index >= 0.0

    def test_fire_weather_index_extreme(self) -> None:
        w = WeatherState(temperature=40.0, humidity=0.1, wind_speed=20.0)
        assert w.fire_weather_index > 5.0

    def test_fire_weather_index_precipitation_suppresses(self) -> None:
        w_dry = WeatherState(temperature=35.0, humidity=0.2, wind_speed=10.0, precipitation=0.0)
        w_wet = WeatherState(temperature=35.0, humidity=0.2, wind_speed=10.0, precipitation=10.0)
        assert w_wet.fire_weather_index < w_dry.fire_weather_index

    def test_wind_vector(self) -> None:
        w = WeatherState(wind_speed=10.0, wind_direction=90.0)
        vec = w.wind_vector
        assert vec.shape == (2,)
        assert vec[0] == pytest.approx(10.0, abs=0.01)
        assert vec[1] == pytest.approx(0.0, abs=0.01)

    def test_wind_vector_north(self) -> None:
        w = WeatherState(wind_speed=5.0, wind_direction=0.0)
        vec = w.wind_vector
        assert vec[0] == pytest.approx(0.0, abs=0.01)
        assert vec[1] == pytest.approx(5.0, abs=0.01)

    def test_moisture_drying_rate_nonneg(self) -> None:
        w = WeatherState()
        rate = w.moisture_drying_rate()
        assert rate >= 0.0

    def test_moisture_drying_rate_bounded(self) -> None:
        w = WeatherState(temperature=45.0, humidity=0.0, wind_speed=30.0)
        assert w.moisture_drying_rate() <= 0.15
