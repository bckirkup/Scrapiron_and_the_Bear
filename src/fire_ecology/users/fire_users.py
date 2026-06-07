"""Factory for fire-ecology user profiles matching the spec §6."""

from __future__ import annotations

import numpy as np
from tattletots.models.user import User


def create_fire_users(n_signal_dims: int = 10) -> list[User]:
    """Create the three fire-domain user profiles.

    Each user has a distinct priority vector reflecting their operational focus:
    - Sector Manager: geographic coverage and hotspot detection.
    - Fire Operations Chief: suppression, dispatch, and resource allocation.
    - Controlled-Burn Manager: prescribed burn monitoring and escape detection.

    Args:
        n_signal_dims: Dimensionality of the signal/priority vectors.

    Returns:
        List of three User objects.
    """
    sector_priority = np.zeros(n_signal_dims)
    sector_priority[: n_signal_dims // 3] = 1.0
    sector_priority /= max(float(np.linalg.norm(sector_priority)), 1e-10)

    ops_priority = np.zeros(n_signal_dims)
    ops_priority[n_signal_dims // 3 : 2 * n_signal_dims // 3] = 1.0
    ops_priority /= max(float(np.linalg.norm(ops_priority)), 1e-10)

    burn_priority = np.zeros(n_signal_dims)
    burn_priority[2 * n_signal_dims // 3 :] = 1.0
    burn_priority /= max(float(np.linalg.norm(burn_priority)), 1e-10)

    return [
        User(
            name="Sector Manager",
            attention_budget=1.0,
            priority_vector=sector_priority,
        ),
        User(
            name="Fire Operations Chief",
            attention_budget=1.5,
            priority_vector=ops_priority,
        ),
        User(
            name="Controlled-Burn Manager",
            attention_budget=0.8,
            priority_vector=burn_priority,
        ),
    ]
