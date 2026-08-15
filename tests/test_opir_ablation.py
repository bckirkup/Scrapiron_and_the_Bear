"""Tests for the OPIR-backstop ablation and the grounded-access measurement arms.

The OPIR backstop used to be appended to A4 detections unconditionally, which
kept domain detection metrics alive even with every Tot dead.  These tests pin
the two paths apart: the assisted path still detects without agents, the
ablated path does not.
"""

from __future__ import annotations

import numpy as np
import pytest
from tattletots.models.agent import LifecycleStage

from fire_ecology.architectures.a4_bma import BMAFireEcology
from fire_ecology.architectures.ecology_options import (
    EcologyMeasurementOptions,
    engine_supports,
)
from fire_ecology.comparison import ComparisonConfig, build_fresh_grid, evolve_weather
from fire_ecology.measurement import (
    ArmSpec,
    SweepSpec,
    reproductive_correlation,
    run_arm,
    sweep_arms,
)
from fire_ecology.sensors.opir import OPIRSatellite

GROUNDED_KNOBS_AVAILABLE = engine_supports("grounded_input_fraction")
requires_grounded_knobs = pytest.mark.skipif(
    not GROUNDED_KNOBS_AVAILABLE,
    reason="installed TattleTots engine has no grounded_input_fraction knob",
)


class _Run:
    """Totals from one short A4 run, split into agent-only and reported."""

    def __init__(self) -> None:
        self.agent_detections = 0
        self.reported_detections = 0
        self.opir_detections = 0
        self.opir_shadow_detections = 0
        self.reports_issued = 0
        self.living_population = 0


def _run_arch(
    *,
    ablate: bool,
    steps: int = 12,
    kill_from: int | None = None,
    seed: int = 7,
    grounded_input_fraction: float = 0.0,
    count_from: int = 0,
) -> _Run:
    """Run A4 over the shared fire physics, optionally killing every Tot."""
    options = EcologyMeasurementOptions(
        ablate_opir_backstop=ablate,
        grounded_input_fraction=grounded_input_fraction,
    )
    config = ComparisonConfig(steps=steps, seed=seed)
    arch = BMAFireEcology(
        n_drones=5, grid_rows=12, grid_cols=12, seed=seed, initial_population=8, options=options
    )
    rng = np.random.default_rng(seed)
    weather_rng = np.random.default_rng(seed + 1)
    grid = build_fresh_grid(config, np.random.default_rng(seed))
    opir = OPIRSatellite(cadence=2)
    totals = _Run()

    for step in range(steps):
        weather = evolve_weather(step, weather_rng, config.weather_volatility)
        grid.step(weather, step, rng)
        grid.stochastic_ignition(weather, step, rng, base_rate=config.base_ignition_rate)
        if kill_from is not None and step >= kill_from and arch.world is not None:
            for agent in arch.world.agents.values():
                agent.state.lifecycle = LifecycleStage.DEAD
        result = arch.step(grid, weather, opir, step, rng)
        if step < count_from:
            continue
        totals.agent_detections += result.tot_detections
        totals.reported_detections += len(result.detections)
        totals.opir_detections += result.opir_detections
        totals.opir_shadow_detections += result.opir_shadow_detections
        totals.reports_issued += result.reports_issued
    totals.living_population = arch.living_population
    return totals


def _thermal_feed(*, ablate: bool, use_opir: bool, seed: int = 5) -> np.ndarray:
    """Thermal stream contents after one step, i.e. what the agents can read."""
    grid = build_fresh_grid(ComparisonConfig(steps=1, seed=seed), np.random.default_rng(seed))
    weather = evolve_weather(0, np.random.default_rng(seed + 1), 0.2)
    arch = BMAFireEcology(
        n_drones=5,
        grid_rows=20,
        grid_cols=20,
        seed=seed,
        initial_population=8,
        use_opir=use_opir,
        options=EcologyMeasurementOptions(ablate_opir_backstop=ablate),
    )
    arch.step(grid, weather, OPIRSatellite(cadence=1), 0, np.random.default_rng(seed))
    assert arch.world is not None
    thermal = next(
        stream for stream in arch.world.streams.values() if stream.label == "thermal_detection"
    )
    return np.asarray(thermal.current_data, dtype=np.float64)


class TestOpirBackstopAblation:
    def test_default_keeps_the_backstop(self) -> None:
        """Default options are the legacy OPIR-assisted path."""
        totals = _run_arch(ablate=False)
        assert totals.opir_detections > 0
        assert totals.opir_shadow_detections == 0
        assert totals.reported_detections == totals.agent_detections + totals.opir_detections

    def test_ablated_path_reports_only_agent_detections(self) -> None:
        """Ablation withholds OPIR hits and books them as shadow detections."""
        totals = _run_arch(ablate=True)
        assert totals.opir_detections == 0
        assert totals.opir_shadow_detections > 0
        assert totals.reported_detections == totals.agent_detections

    def test_ablation_bites_when_agents_are_extinct(self) -> None:
        """With every Tot dead the ablated path detects nothing; assisted still does."""
        ablated = _run_arch(ablate=True, kill_from=3, count_from=3)
        assisted = _run_arch(ablate=False, kill_from=3, count_from=3)

        assert ablated.living_population == 0
        assert assisted.living_population == 0
        assert ablated.reports_issued == 0
        assert ablated.agent_detections == 0
        assert ablated.reported_detections == 0
        # The masking this ablation removes: OPIR alone carries the metric.
        assert assisted.reported_detections > 0
        assert assisted.agent_detections == 0

    def test_opir_sensor_stays_wired_when_backstop_is_ablated(self) -> None:
        """Ablating the backstop must not silence the OPIR sensor feed."""
        ablated = _thermal_feed(ablate=True, use_opir=True)
        assisted = _thermal_feed(ablate=False, use_opir=True)
        no_opir = _thermal_feed(ablate=True, use_opir=False)

        np.testing.assert_allclose(ablated, assisted)
        assert float(np.abs(ablated).sum()) > 0.0
        assert not np.allclose(ablated, no_opir)

    def test_disabling_opir_entirely_leaves_no_shadow(self) -> None:
        """``use_opir=False`` removes the sensor, so there is nothing to withhold."""
        grid = build_fresh_grid(ComparisonConfig(steps=1, seed=5), np.random.default_rng(5))
        weather = evolve_weather(0, np.random.default_rng(6), 0.2)
        arch = BMAFireEcology(
            n_drones=5,
            grid_rows=20,
            grid_cols=20,
            seed=5,
            initial_population=8,
            use_opir=False,
            options=EcologyMeasurementOptions(ablate_opir_backstop=True),
        )
        result = arch.step(grid, weather, OPIRSatellite(cadence=1), 0, np.random.default_rng(5))

        assert result.opir_detections == 0
        assert result.opir_shadow_detections == 0

    def test_first_step_agent_detections_match_across_paths(self) -> None:
        """Before suppression diverges the two paths see the same agent behavior."""
        ablated = _run_arch(ablate=True, steps=1)
        assisted = _run_arch(ablate=False, steps=1)
        assert ablated.agent_detections == assisted.agent_detections
        assert ablated.reports_issued == assisted.reports_issued
        assert ablated.opir_shadow_detections == assisted.opir_detections


class TestEcologyMeasurementOptions:
    def test_defaults_forward_no_engine_kwargs(self) -> None:
        assert EcologyMeasurementOptions().engine_kwargs() == {}

    @requires_grounded_knobs
    @pytest.mark.parametrize("fraction", [0.0, 0.34, 0.67])
    def test_grounded_fraction_is_forwarded_when_non_default(self, fraction: float) -> None:
        kwargs = EcologyMeasurementOptions(grounded_input_fraction=fraction).engine_kwargs()
        if fraction == 0.0:
            assert "grounded_input_fraction" not in kwargs
        else:
            assert kwargs["grounded_input_fraction"] == pytest.approx(fraction)

    def test_unsupported_knob_raises_instead_of_measuring_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "fire_ecology.architectures.ecology_options.engine_supports", lambda _name: False
        )
        with pytest.raises(ValueError, match="grounded_input_fraction"):
            EcologyMeasurementOptions(grounded_input_fraction=0.67).engine_kwargs()
        assert EcologyMeasurementOptions().engine_kwargs() == {}


class TestReproductiveCorrelation:
    def test_too_few_pairs_is_undefined(self) -> None:
        assert reproductive_correlation({"a": [], "b": ["a"]}) is None

    def test_zero_variance_is_undefined(self) -> None:
        parents = {"root": [], "a": ["root"], "b": ["root"], "c": ["root"]}
        assert reproductive_correlation(parents) is None

    def test_perfectly_matched_offspring_counts_correlate(self) -> None:
        # Each parent has as many children as its own parent did.
        parents: dict[str, list[str]] = {"r": []}
        parents.update({f"g1_{i}": ["r"] for i in range(3)})
        for i in range(3):
            parents.update({f"g2_{i}_{j}": [f"g1_{i}"] for j in range(i + 1)})
        value = reproductive_correlation(parents)
        assert value is not None
        assert -1.0 <= value <= 1.0


@requires_grounded_knobs
class TestGroundedAccessArms:
    @pytest.mark.parametrize("fraction", [0.0, 0.34, 0.67])
    def test_arm_metrics_stay_in_bounds(self, fraction: float) -> None:
        sweep = SweepSpec(steps=15, grid_rows=12, grid_cols=12, seed=11, instrument_steps=15)
        spec = ArmSpec(label=f"gif{fraction:.2f}", grounded_input_fraction=fraction)
        numbers = run_arm(spec, sweep).key_numbers()

        for key in (
            "agent_only_detection_rate",
            "agent_only_step_detection_rate",
            "agent_only_cell_recall",
            "report_precision",
            "report_false_alarm_rate",
            "null_chance_precision",
            "null_static_prior_precision",
            "attention_solvent_fraction",
            "grounded_yield_share",
        ):
            assert 0.0 <= numbers[key] <= 1.0, key
        assert numbers["opir_backstop_ablated"] is True
        assert numbers["reported_detection_rate"] >= numbers["agent_only_detection_rate"]
        assert numbers["mean_per_capita_attention_capacity"] >= 0.0

    def test_grounded_access_grades_the_yield_share(self) -> None:
        """More reserved raw slots must not lower the grounded yield share."""
        sweep = SweepSpec(steps=15, grid_rows=12, grid_cols=12, seed=11, instrument_steps=15)
        shares = [
            run_arm(ArmSpec(label=f"gif{f:.2f}", grounded_input_fraction=f), sweep).key_numbers()[
                "grounded_yield_share"
            ]
            for f in (0.0, 0.34, 0.67)
        ]
        assert shares[0] <= shares[1] <= shares[2]
        assert shares[2] > shares[0]

    def test_sweep_arms_pairs_each_fraction_with_its_assisted_twin(self) -> None:
        arms = sweep_arms((0.0, 0.34))
        assert [arm.label for arm in arms] == [
            "gif0.00_opir_ablated",
            "gif0.34_opir_ablated",
            "gif0.00_opir_assisted",
            "gif0.34_opir_assisted",
        ]
        assert [arm.ablate_opir_backstop for arm in arms] == [True, True, False, False]

    def test_output_json_round_trips_with_nulls(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        sweep = SweepSpec(steps=10, grid_rows=10, grid_cols=10, seed=3, instrument_steps=10)
        result = run_arm(ArmSpec(label="gif0.34", grounded_input_fraction=0.34), sweep)
        path = tmp_path / "arm.json"
        result.output.write_json(path)

        assert path.exists()
        nulls = result.output.domain_metrics["instrument_nulls"]
        assert "null_chance_baseline" in nulls
        assert "null_static_prior_baseline" in nulls
        assert result.output.run_summary.domain == "fire_ecology"
        assert result.output.run_summary.steps_completed == 10
