"""Competing architectures (A0-A4) for fire-regime management comparison."""

from __future__ import annotations

from fire_ecology.architectures.a0_human import HumanBaseline
from fire_ecology.architectures.a1_camera_ml import CameraMLNetwork
from fire_ecology.architectures.a2_centralized import CentralizedOptimizer
from fire_ecology.architectures.a3_federated import FederatedEdge
from fire_ecology.architectures.a4_bma import BMAFireEcology
from fire_ecology.architectures.base import Architecture, ArchitectureResult

__all__ = [
    "Architecture",
    "ArchitectureResult",
    "BMAFireEcology",
    "CameraMLNetwork",
    "CentralizedOptimizer",
    "FederatedEdge",
    "HumanBaseline",
]
