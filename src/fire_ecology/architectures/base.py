"""Abstract base class for competing fire-management architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field

from fire_ecology.environment.fire import FireGrid
from fire_ecology.environment.weather import WeatherState
from fire_ecology.sensors.opir import OPIRSatellite


class ArchitectureResult(BaseModel):
    """Result of a single step from a competing architecture."""

    detections: list[tuple[int, int]] = Field(
        default_factory=list, description="Fire cells detected this step"
    )
    suppressions: list[tuple[int, int]] = Field(
        default_factory=list, description="Fire cells targeted for suppression"
    )
    escalations: int = Field(default=0, ge=0, description="Escalations to human operators")
    cost: float = Field(default=0.0, ge=0.0, description="Operational cost this step")
    tot_detections: int = Field(default=0, ge=0, description="Detections attributed to Tot reports")
    opir_detections: int = Field(
        default=0, ge=0, description="Detections attributed to the OPIR backstop"
    )
    living_population: int | None = Field(
        default=None, ge=0, description="Living Tots after this step, when applicable"
    )
    ecology_extinct: bool = Field(
        default=False, description="Whether the Tot ecology has reached zero living agents"
    )

    model_config = {"arbitrary_types_allowed": True}


class Architecture(ABC):
    """Base class for competing fire-management architectures.

    Each architecture receives the same sensor feeds and hardware availability.
    The step() method returns what the architecture detected, suppressed,
    and escalated, along with operational cost.
    """

    @abstractmethod
    def step(
        self,
        fire_grid: FireGrid,
        weather: WeatherState,
        opir: OPIRSatellite,
        time_step: int,
        rng: np.random.Generator,
    ) -> ArchitectureResult:
        """Execute one step of the architecture's fire management logic."""

    @abstractmethod
    def reset(self) -> None:
        """Reset architecture state for a new simulation run."""
