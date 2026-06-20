"""Tests for fire_ecology.runner (domain-runner integration)."""

from __future__ import annotations

import pytest
from domain_runner.layer import DomainOnlyLayer
from domain_runner.single import run_simulation
from domain_runner.types import RunContext

from fire_ecology.runner import FireDomainHooks, run_fire_simulation


@pytest.mark.integration
class TestFireRunner:
    def test_domain_only_simulation(self) -> None:
        run = RunContext(
            steps=3,
            seed=42,
            domain_config={"grid_rows": 8, "grid_cols": 8, "n_drones": 0},
            layer="domain_only",
        )
        result = run_simulation(FireDomainHooks(), DomainOnlyLayer(), run)
        assert result.steps_completed == 3
        assert result.layer == "domain_only"
        assert "total_burned_area" in result.domain_metrics

    def test_load_run_context_from_cli_overrides(self) -> None:
        hooks = FireDomainHooks()
        run = hooks.load_run_context(
            cli_overrides={
                "domain": {"steps": 5, "grid_rows": 6, "grid_cols": 6},
                "layer": "domain_only",
            }
        )
        assert run.steps == 5
        assert run.layer == "domain_only"

    @pytest.mark.smoke
    def test_run_fire_simulation_entry(self) -> None:
        hooks = FireDomainHooks()
        run = hooks.load_run_context(
            cli_overrides={
                "domain": {"steps": 10, "grid_rows": 10, "grid_cols": 10},
                "layer": "domain_only",
            }
        )
        result = run_fire_simulation(run)
        assert result.domain == "fire_ecology"

    def test_tattletots_layer_short(self) -> None:
        hooks = FireDomainHooks()
        run = hooks.load_run_context(
            cli_overrides={
                "domain": {"steps": 8, "grid_rows": 8, "grid_cols": 8},
                "layer": "tattletots",
                "simulation": {"initial_population": 8, "max_steps": 8, "seed": 42},
            }
        )
        result = run_fire_simulation(run)
        assert result.layer == "tattletots"
        assert result.steps_completed == 8
