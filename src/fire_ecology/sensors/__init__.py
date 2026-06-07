"""Sensor models: OPIR, camera towers, weather stations, fuel moisture probes."""

from __future__ import annotations

from fire_ecology.sensors.camera_tower import CameraTower
from fire_ecology.sensors.fuel_moisture import FuelMoistureSensor
from fire_ecology.sensors.opir import OPIRSatellite
from fire_ecology.sensors.weather_station import WeatherStation

__all__ = [
    "CameraTower",
    "FuelMoistureSensor",
    "OPIRSatellite",
    "WeatherStation",
]
