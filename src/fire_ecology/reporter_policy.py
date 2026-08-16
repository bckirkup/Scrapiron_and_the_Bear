"""Hand-designed, evidence-only reporter for the fire domain's public streams.

The policy reads only what the adapter publishes to every agent through
:class:`tattletots.interface.reporter_policy.ReporterStream`: the sampled
``thermal_detection`` values, their per-feature observation status, and the
declared coordinates of the observed features. It never touches the
:class:`~fire_ecology.environment.fire.FireGrid`, ``get_active_locations`` or any
adapter internal, so it is a detector a real operator could build.

Its one piece of hand-coded competence is a confidence floor. Both fire sensors
stamp a confidence on every detection they publish: a camera tower reports at
least ``0.6`` and only for cells it actually sees burning, OPIR reports ``0.5``
or more for a true hit and exactly ``0.3`` for a false positive on an unburned
cell. A reporter that ignores the published confidence therefore inherits the
OPIR false-positive rate; a reporter that requires a confidence above the
false-positive stamp does not. The floor is a property of the reporter, not of
the domain: nothing in the fire model is tuned for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tattletots.interface.reporter_policy import (
    ReporterDecision,
    ReporterPolicyContext,
    ReporterStream,
    register_reporter_policy,
)
from tattletots.models.location import EventLocation, LocationFrame

FIRE_REPORTER_POLICY_NAME = "fire_thermal_evidence"
THERMAL_STREAM_LABEL = "thermal_detection"
OBSERVED_STATUS = "observed"

#: Above the ``0.3`` confidence OPIR stamps on a false positive and below the
#: ``0.5`` floor of a true OPIR hit or the ``0.6`` floor of a camera detection.
DEFAULT_CONFIDENCE_FLOOR = 0.45


@dataclass
class FireThermalEvidenceReporterPolicy:
    """Report the strongest observed thermal detection that clears the floor.

    The counters are measurement instrumentation: ``decision_steps`` counts the
    adult steps on which the engine consulted this policy, and the evidence
    counters count how many of those steps carried usable public evidence.
    """

    confidence_floor: float = DEFAULT_CONFIDENCE_FLOOR
    decision_steps: int = 0
    thermal_stream_steps: int = 0
    thermal_observed_steps: int = 0
    thermal_evidence_steps: int = 0
    escalations: int = 0

    def decide(self, context: ReporterPolicyContext) -> ReporterDecision:
        """Escalate only when a published thermal detection supports it."""
        self.decision_steps += 1
        stream = self._find_stream(context.streams, THERMAL_STREAM_LABEL)
        if stream is None or not self._has_declarations(stream):
            return ReporterDecision(escalate=False)
        self.thermal_stream_steps += 1

        observed = self._observed_features(stream)
        if observed:
            self.thermal_observed_steps += 1

        location = self._strongest_detection(observed)
        if location is None or not self._in_frame(location, context.location_frame):
            return ReporterDecision(escalate=False)

        self.thermal_evidence_steps += 1
        self.escalations += 1
        return ReporterDecision(escalate=True, location=location)

    @staticmethod
    def _find_stream(
        streams: tuple[ReporterStream, ...],
        label: str,
    ) -> ReporterStream | None:
        return next((stream for stream in streams if stream.label == label), None)

    @staticmethod
    def _has_declarations(stream: ReporterStream) -> bool:
        coordinates = stream.metadata.coordinates
        if coordinates is None:
            return False
        if len(coordinates) != stream.data.size:
            return False
        return len(stream.observation_status) == stream.data.size

    @staticmethod
    def _observed_features(stream: ReporterStream) -> list[tuple[float, EventLocation]]:
        """Values and declared cells of the features the domain says are observed."""
        coordinates = stream.metadata.coordinates
        if coordinates is None:
            return []
        features: list[tuple[float, EventLocation]] = []
        for value, status, coordinate in zip(
            stream.data,
            stream.observation_status,
            coordinates,
            strict=True,
        ):
            if status != OBSERVED_STATUS or coordinate is None or len(coordinate) < 2:
                continue
            if not np.isfinite(value):
                continue
            features.append((float(value), (int(round(coordinate[0])), int(round(coordinate[1])))))
        return features

    def _strongest_detection(
        self,
        features: list[tuple[float, EventLocation]],
    ) -> EventLocation | None:
        """Cell of the strongest detection above the confidence floor, if any."""
        cleared = [feature for feature in features if feature[0] > self.confidence_floor]
        if not cleared:
            return None
        return max(cleared, key=lambda feature: feature[0])[1]

    @staticmethod
    def _in_frame(location: EventLocation, frame: LocationFrame | None) -> bool:
        if frame is None:
            return True
        minimum, maximum = frame
        if not minimum[0] <= location[0] <= maximum[0]:
            return False
        return minimum[1] <= location[1] <= maximum[1]


def make_fire_reporter_policy() -> FireThermalEvidenceReporterPolicy:
    """Factory used by the engine's reporter-policy registry."""
    return FireThermalEvidenceReporterPolicy()


register_reporter_policy(FIRE_REPORTER_POLICY_NAME, make_fire_reporter_policy)
