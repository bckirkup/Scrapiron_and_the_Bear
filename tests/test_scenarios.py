"""Tests for deployment scenarios."""

from __future__ import annotations

from fire_ecology.scenarios.phased_deployment import (
    DeploymentPhase,
    PhasedDeploymentConfig,
)


class TestPhasedDeployment:
    def test_default_four_phases(self) -> None:
        config = PhasedDeploymentConfig()
        assert len(config.phases) == 4

    def test_phase_0_at_start(self) -> None:
        config = PhasedDeploymentConfig(steps_per_phase=100)
        assert config.get_phase(0) == DeploymentPhase.PHASE_0
        assert config.get_phase(50) == DeploymentPhase.PHASE_0

    def test_phase_progression(self) -> None:
        config = PhasedDeploymentConfig(steps_per_phase=100)
        assert config.get_phase(0) == DeploymentPhase.PHASE_0
        assert config.get_phase(100) == DeploymentPhase.PHASE_1
        assert config.get_phase(200) == DeploymentPhase.PHASE_2
        assert config.get_phase(300) == DeploymentPhase.PHASE_3

    def test_phase_3_caps(self) -> None:
        config = PhasedDeploymentConfig(steps_per_phase=100)
        assert config.get_phase(999) == DeploymentPhase.PHASE_3

    def test_get_config_returns_correct_drones(self) -> None:
        config = PhasedDeploymentConfig(steps_per_phase=100)
        p0 = config.get_config(0)
        p1 = config.get_config(100)
        assert p0.n_drones == 5
        assert p1.n_drones == 15

    def test_total_steps(self) -> None:
        config = PhasedDeploymentConfig(steps_per_phase=100)
        assert config.total_steps == 400

    def test_opir_always_available(self) -> None:
        config = PhasedDeploymentConfig()
        for phase_config in config.phases.values():
            assert phase_config.has_opir

    def test_autonomous_suppression_phase_progression(self) -> None:
        config = PhasedDeploymentConfig()
        assert not config.phases[DeploymentPhase.PHASE_0].autonomous_suppression
        assert config.phases[DeploymentPhase.PHASE_1].autonomous_suppression
