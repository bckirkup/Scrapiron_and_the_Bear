"""Measurement options for the A4 ecology arm.

Two independent things used to be entangled in ``use_opir``:

* whether the OPIR satellite feeds the agents' thermal stream, and
* whether OPIR detections are appended to the architecture's detections.

The second one masks the initiation problem: the OPIR backstop keeps domain
detection metrics alive even when every Tot is dead, so nothing measured
through that path says whether the agents contribute anything. ``ablate_opir_backstop``
removes the backstop from the reported detections while leaving the sensor feed
and its random draws intact, so an ablated arm is directly comparable to its
baseline at the same seed.

The grounded-access fields forward the TattleTots
``grounded_input_fraction`` / ``grounded_attractiveness_multiplier`` /
``max_input_streams`` knobs. They are forwarded only when the installed engine
declares them, so a pinned engine without those fields still runs the default
arm; requesting a non-default value against such an engine raises instead of
silently measuring the legacy behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tattletots.engine.config import SimulationConfig

GROUNDED_ENGINE_DEFAULTS: dict[str, float] = {
    "grounded_input_fraction": 0.0,
    "grounded_attractiveness_multiplier": 1.0,
    "max_input_streams": 3.0,
}


def engine_supports(field_name: str) -> bool:
    """Whether the installed TattleTots ``SimulationConfig`` declares ``field_name``."""
    return field_name in SimulationConfig.model_fields


@dataclass(frozen=True)
class EcologyMeasurementOptions:
    """Measurement-path options for :class:`~fire_ecology.architectures.a4_bma.BMAFireEcology`.

    All defaults reproduce the historical behavior exactly, including random
    number consumption.
    """

    ablate_opir_backstop: bool = False
    grounded_input_fraction: float = 0.0
    grounded_attractiveness_multiplier: float = 1.0
    max_input_streams: int = 3

    def grounded_values(self) -> dict[str, float]:
        """Requested engine values for the grounded-access knobs."""
        return {
            "grounded_input_fraction": self.grounded_input_fraction,
            "grounded_attractiveness_multiplier": self.grounded_attractiveness_multiplier,
            "max_input_streams": float(self.max_input_streams),
        }

    def engine_kwargs(self) -> dict[str, Any]:
        """Return ``SimulationConfig`` keyword arguments for the requested knobs.

        Raises:
            ValueError: a non-default value was requested for a knob the
                installed engine does not declare.
        """
        kwargs: dict[str, Any] = {}
        for name, value in self.grounded_values().items():
            is_default = math.isclose(value, GROUNDED_ENGINE_DEFAULTS[name])
            if not engine_supports(name):
                if not is_default:
                    raise ValueError(
                        f"The installed TattleTots engine has no {name!r} field, so the "
                        f"requested value {value!r} cannot be measured. Install an engine "
                        "build that declares the grounded raw-stream access knobs."
                    )
                continue
            if not is_default:
                kwargs[name] = int(value) if name == "max_input_streams" else value
        return kwargs

    def as_dict(self) -> dict[str, Any]:
        """Serializable description of this arm's options."""
        return {
            "ablate_opir_backstop": self.ablate_opir_backstop,
            "grounded_input_fraction": self.grounded_input_fraction,
            "grounded_attractiveness_multiplier": self.grounded_attractiveness_multiplier,
            "max_input_streams": self.max_input_streams,
        }
