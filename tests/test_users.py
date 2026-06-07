"""Tests for fire-ecology user profiles."""

from __future__ import annotations

import numpy as np

from fire_ecology.users.fire_users import create_fire_users


class TestFireUsers:
    def test_creates_three_users(self) -> None:
        users = create_fire_users()
        assert len(users) == 3

    def test_user_names(self) -> None:
        users = create_fire_users()
        names = [u.name for u in users]
        assert "Sector Manager" in names
        assert "Fire Operations Chief" in names
        assert "Controlled-Burn Manager" in names

    def test_priority_vectors_normalized(self) -> None:
        users = create_fire_users(n_signal_dims=12)
        for user in users:
            norm = float(np.linalg.norm(user.priority_vector))
            assert abs(norm - 1.0) < 0.01

    def test_priority_vectors_orthogonal(self) -> None:
        users = create_fire_users(n_signal_dims=12)
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                dot = float(np.dot(users[i].priority_vector, users[j].priority_vector))
                assert abs(dot) < 0.01

    def test_attention_budgets(self) -> None:
        users = create_fire_users()
        for user in users:
            assert user.attention_budget > 0.0

    def test_ops_chief_highest_budget(self) -> None:
        users = create_fire_users()
        ops = [u for u in users if u.name == "Fire Operations Chief"][0]
        assert ops.attention_budget >= max(u.attention_budget for u in users if u != ops)
