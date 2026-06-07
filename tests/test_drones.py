"""Tests for drone body plans, genome, and state."""

from __future__ import annotations

import numpy as np

from fire_ecology.drones.body_plan import BodyPlan, BodyPlanType
from fire_ecology.drones.drone_genome import DroneGenome
from fire_ecology.drones.drone_state import DroneState


class TestBodyPlan:
    def test_scout_factory(self) -> None:
        bp = BodyPlan.scout()
        assert bp.plan_type == BodyPlanType.SCOUT
        assert bp.tank_gallons == 0.0
        assert bp.sensor_count >= 4
        assert not bp.can_suppress

    def test_strike_small_factory(self) -> None:
        bp = BodyPlan.strike_small()
        assert bp.plan_type == BodyPlanType.STRIKE_SMALL
        assert bp.tank_gallons == 5.0
        assert bp.can_suppress

    def test_strike_large_factory(self) -> None:
        bp = BodyPlan.strike_large()
        assert bp.plan_type == BodyPlanType.STRIKE_LARGE
        assert bp.tank_gallons == 40.0

    def test_relay_factory(self) -> None:
        bp = BodyPlan.relay()
        assert bp.plan_type == BodyPlanType.RELAY
        assert bp.endurance > 80

    def test_hybrid_factory(self) -> None:
        bp = BodyPlan.hybrid()
        assert bp.plan_type == BodyPlanType.HYBRID
        assert bp.can_suppress

    def test_suppression_effectiveness_scales_with_tank(self) -> None:
        scout = BodyPlan.scout()
        small = BodyPlan.strike_small()
        large = BodyPlan.strike_large()
        assert scout.suppression_effectiveness == 0.0
        assert 0.2 < small.suppression_effectiveness < 0.5
        assert large.suppression_effectiveness > small.suppression_effectiveness
        assert large.suppression_effectiveness <= 0.95

    def test_suppression_effectiveness_no_tank(self) -> None:
        bp = BodyPlan(tank_gallons=0.0)
        assert bp.suppression_effectiveness == 0.0


class TestDroneGenome:
    def test_default_fractions_sum_to_one(self) -> None:
        genome = DroneGenome()
        total = genome.patrol_fraction + genome.suppress_fraction + genome.report_fraction
        assert abs(total - 1.0) < 0.01

    def test_expected_role_scout(self) -> None:
        genome = DroneGenome(
            body_plan=BodyPlan.scout(),
            patrol_fraction=0.6,
            suppress_fraction=0.1,
            report_fraction=0.3,
        )
        assert genome.expected_role == "scout"

    def test_expected_role_strike(self) -> None:
        genome = DroneGenome(
            body_plan=BodyPlan.strike_large(),
            suppress_fraction=0.5,
            patrol_fraction=0.2,
            report_fraction=0.3,
        )
        assert genome.expected_role == "strike"

    def test_expected_role_relay(self) -> None:
        genome = DroneGenome(body_plan=BodyPlan.relay())
        assert genome.expected_role == "relay"

    def test_expected_role_broker(self) -> None:
        genome = DroneGenome(
            body_plan=BodyPlan.scout(),
            report_fraction=0.6,
            patrol_fraction=0.2,
            suppress_fraction=0.2,
        )
        assert genome.expected_role == "broker"

    def test_mutate_preserves_fraction_sum(self) -> None:
        rng = np.random.default_rng(42)
        genome = DroneGenome()
        mutated = genome.mutate(rng, rate=0.5)
        total = mutated.patrol_fraction + mutated.suppress_fraction + mutated.report_fraction
        assert abs(total - 1.0) < 0.01

    def test_mutate_preserves_body_plan(self) -> None:
        rng = np.random.default_rng(42)
        genome = DroneGenome(body_plan=BodyPlan.scout())
        mutated = genome.mutate(rng, rate=1.0)
        assert mutated.body_plan.plan_type == BodyPlanType.SCOUT

    def test_mutate_changes_values(self) -> None:
        rng = np.random.default_rng(42)
        genome = DroneGenome()
        mutated = genome.mutate(rng, rate=1.0)
        changed = (
            mutated.preferred_terrain_risk != genome.preferred_terrain_risk
            or mutated.escalation_to_human != genome.escalation_to_human
            or mutated.swap_threshold != genome.swap_threshold
        )
        assert changed


class TestDroneState:
    def test_default_fully_charged(self) -> None:
        state = DroneState()
        assert state.battery == 1.0
        assert state.water == 1.0
        assert not state.needs_swap
        assert not state.needs_water

    def test_needs_swap_low_battery(self) -> None:
        state = DroneState(battery=0.05)
        assert state.needs_swap

    def test_needs_water_empty(self) -> None:
        state = DroneState(water=0.0)
        assert state.needs_water
