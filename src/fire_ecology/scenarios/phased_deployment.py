"""Phased deployment configuration matching spec §8.

Models the spiral acquisition schedule where hardware is incrementally
deployed over 24 months (mapped to simulation steps).
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class DeploymentPhase(enum.StrEnum):
    """Deployment phases from spec §8."""

    PHASE_0 = "phase_0"
    PHASE_1 = "phase_1"
    PHASE_2 = "phase_2"
    PHASE_3 = "phase_3"


class PhaseConfig(BaseModel):
    """Hardware configuration for a single deployment phase."""

    n_drones: int = Field(ge=0, description="Number of operational drones")
    n_cameras: int = Field(ge=0, description="Number of camera towers")
    n_weather_stations: int = Field(ge=0, description="Number of weather stations")
    n_fuel_sensors: int = Field(ge=0, description="Number of fuel moisture sensors")
    coverage_km2: float = Field(ge=0.0, description="Operational coverage area in km²")
    has_opir: bool = Field(default=True, description="Whether OPIR backstop is available")
    autonomous_suppression: bool = Field(
        default=False, description="Whether autonomous suppression is enabled"
    )


class PhasedDeploymentConfig(BaseModel):
    """Full phased deployment configuration.

    Maps deployment phases to simulation step ranges and hardware configs.

    Default phases (from spec §8):
    - Phase 0 (months 0-6):  5 drones, 3 cameras, 4 weather stations, 50 km²
    - Phase 1 (months 6-12): 15 drones, 8 cameras, expanded sector
    - Phase 2 (months 12-18): 30 drones, 12 cameras, fuel moisture sensors
    - Phase 3 (months 18-24): expansion or fleet adjustment
    """

    steps_per_phase: int = Field(
        default=100,
        ge=1,
        description="Number of simulation steps per phase",
    )
    phases: dict[DeploymentPhase, PhaseConfig] = Field(
        default_factory=lambda: {
            DeploymentPhase.PHASE_0: PhaseConfig(
                n_drones=5,
                n_cameras=3,
                n_weather_stations=4,
                n_fuel_sensors=0,
                coverage_km2=50.0,
                autonomous_suppression=False,
            ),
            DeploymentPhase.PHASE_1: PhaseConfig(
                n_drones=15,
                n_cameras=8,
                n_weather_stations=4,
                n_fuel_sensors=0,
                coverage_km2=100.0,
                autonomous_suppression=True,
            ),
            DeploymentPhase.PHASE_2: PhaseConfig(
                n_drones=30,
                n_cameras=12,
                n_weather_stations=6,
                n_fuel_sensors=4,
                coverage_km2=200.0,
                autonomous_suppression=True,
            ),
            DeploymentPhase.PHASE_3: PhaseConfig(
                n_drones=30,
                n_cameras=12,
                n_weather_stations=8,
                n_fuel_sensors=6,
                coverage_km2=300.0,
                autonomous_suppression=True,
            ),
        }
    )

    def get_phase(self, time_step: int) -> DeploymentPhase:
        """Determine the deployment phase for a given time step."""
        phase_idx = min(time_step // self.steps_per_phase, 3)
        return DeploymentPhase(list(DeploymentPhase)[phase_idx])

    def get_config(self, time_step: int) -> PhaseConfig:
        """Get the hardware config for a given time step."""
        return self.phases[self.get_phase(time_step)]

    @property
    def total_steps(self) -> int:
        """Total simulation steps across all phases."""
        return self.steps_per_phase * len(self.phases)
