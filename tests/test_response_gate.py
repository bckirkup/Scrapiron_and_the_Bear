"""Tests for the response-gate (lever 5) measurement.

Two things are claimed and are tested separately: that the payoff levers are
config-gated, off by default, and translated into exactly the engine knobs the
measurement says it sets, with the correctness weight the only quantity that
differs between arms; and that the harness turns per-seed ledger metrics into
pooled arm summaries and clause verdicts that respond to their inputs. Golden
numbers are avoided in favour of bounds, ordering and graded sensitivity.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from fire_ecology.architectures.ecology_options import engine_supports
from fire_ecology.measurement import response_gate as rg
from fire_ecology.measurement.designed_reporter import (
    ALL_DESIGNED_ARM,
    ORDINARY_ARM,
    DesignedReporterSpec,
    PayoffLevers,
    SeedRun,
    build_world,
)

requires_grounded_knobs = pytest.mark.skipif(
    not engine_supports("grounded_input_fraction"),
    reason="installed TattleTots engine has no grounded_input_fraction knob",
)

_SMALL_SPEC = DesignedReporterSpec(
    steps=30,
    grid_rows=12,
    grid_cols=12,
    n_cameras=2,
    base_ignition_rate=0.01,
    initial_population=6,
    max_population=12,
    levers=PayoffLevers(enabled=True),
)


def test_levers_are_off_by_default() -> None:
    """The committed numbers must be reproducible: no knobs unless asked."""
    levers = PayoffLevers()
    assert levers.enabled is False
    assert levers.engine_kwargs() == {}
    assert levers.as_dict() == {}
    assert DesignedReporterSpec().as_dict().get("payoff_levers") is None


def test_enabled_levers_set_exactly_the_measured_knobs() -> None:
    kwargs = PayoffLevers(enabled=True, reproduction_correctness_weight=1.0).engine_kwargs()
    assert kwargs == {
        "correct_report_attention_value": 8.0,
        "reproduction_merit_ordering": True,
        "escalation_calibration_in_score_units": True,
        "false_alarm_break_even_precision": 0.2,
        "reproduction_correctness_weight": 1.0,
    }


def test_engine_accepts_the_lever_knobs() -> None:
    """Every knob the levers set is a declared field of the installed engine config.

    ``SimulationConfig`` ignores unknown keyword arguments, so a knob that is not a
    declared field would be silently dropped rather than raising.
    """
    for knob in PayoffLevers(enabled=True).engine_kwargs():
        assert engine_supports(knob), knob
    assert not engine_supports("gene_pool")


def test_gene_pool_carries_the_starting_threshold_range() -> None:
    """The threshold range is a World argument, not a config field."""
    assert PayoffLevers().gene_pool() is None
    pool = PayoffLevers(enabled=True, escalation_threshold_range=(0.05, 0.3)).gene_pool()
    assert pool is not None
    assert pool.escalation_threshold_range == (0.05, 0.3)


@requires_grounded_knobs
def test_built_world_starts_inside_the_requested_threshold_range() -> None:
    _, world = build_world(_SMALL_SPEC, seed=11, arm=ORDINARY_ARM)
    low, high = _SMALL_SPEC.levers.escalation_threshold_range
    assert world.gene_pool is not None
    assert world.gene_pool.escalation_threshold_range == (low, high)
    thresholds = [agent.genome.escalation_threshold for agent in world.agents.values()]
    assert thresholds
    assert all(low <= value <= high for value in thresholds)


def test_only_the_correctness_weight_differs_between_arms() -> None:
    arms = rg.gate_arms((0.0, 1.0))
    evolved = [arm for arm in arms if arm.policy_arm == ORDINARY_ARM]
    kwargs = [arm.spec(_SMALL_SPEC).levers.engine_kwargs() for arm in evolved]
    differing = {key for key in kwargs[0] if kwargs[0][key] != kwargs[1][key]}
    assert differing == {"reproduction_correctness_weight"}
    assert [arm.correctness_weight for arm in evolved] == [0.0, 1.0]


def test_arm_spec_changes_nothing_but_the_levers() -> None:
    arm = rg.GateArm(name="evolved_w1", policy_arm=ORDINARY_ARM, correctness_weight=1.0)
    derived = arm.spec(_SMALL_SPEC)
    assert dataclasses.replace(derived, levers=_SMALL_SPEC.levers) == _SMALL_SPEC
    assert derived.levers.enabled is True
    assert derived.levers.reproduction_correctness_weight == pytest.approx(1.0)


@pytest.mark.parametrize("weights", [(0.0, 1.0), (0.0, 0.5, 1.0), (0.25,)])
def test_gate_arms_cover_the_weights_plus_one_designed_ceiling(weights: tuple[float, ...]) -> None:
    arms = rg.gate_arms(weights)
    evolved = [arm for arm in arms if arm.policy_arm == ORDINARY_ARM]
    designed = [arm for arm in arms if arm.policy_arm == ALL_DESIGNED_ARM]
    assert [arm.correctness_weight for arm in evolved] == sorted(weights)
    assert len(designed) == 1
    assert designed[0].correctness_weight == max(weights)
    assert len({arm.name for arm in arms}) == len(arms)


def test_seed_blocks_are_disjoint() -> None:
    assert not set(rg.PRIMARY_SEEDS) & set(rg.HOLDOUT_SEEDS)
    assert len(rg.PRIMARY_SEEDS) >= 20
    assert len(rg.HOLDOUT_SEEDS) >= 20


def _metrics(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "seed": 1,
        "reports": 100,
        "correct_reports": 30,
        "precision": 0.3,
        "reports_per_adult_lifetime": 1.5,
        "silent_adult_share": 0.4,
        "clause_1_slope": 0.001,
        "generations_observed": 5,
        "clause_2_correlation": 0.05,
        "n_parent_child_pairs": 40,
        "population_capped_step_share": 0.5,
        "eligible_share": 0.66,
        "correct_group_mean_offspring": 1.2,
        "never_correct_group_mean_offspring": 1.1,
        "silent_mean_offspring": 1.0,
        "corr_parent_child_precision": 0.1,
        "final_population": 60,
    }
    row.update(overrides)
    return row


def _arm(name: str = "evolved_w1", weight: float = 1.0) -> rg.GateArm:
    return rg.GateArm(name=name, policy_arm=ORDINARY_ARM, correctness_weight=weight)


@pytest.mark.parametrize("rising", [0, 1, 3, 4])
def test_rising_seed_count_tracks_the_slopes(rising: int) -> None:
    slopes = [0.002] * rising + [-0.002] * (4 - rising)
    summary = rg.summarize_arm(_arm(), [_metrics(clause_1_slope=slope) for slope in slopes])
    assert summary["clause_1_seeds_rising"] == rising
    assert summary["n_seeds"] == 4
    assert summary["mean_clause_1_slope"] == pytest.approx(float(np.mean(slopes)))


@pytest.mark.parametrize("correlation", [-0.5, 0.0, 0.19, 0.21, 0.9])
def test_clause_2_count_uses_the_threshold(correlation: float) -> None:
    summary = rg.summarize_arm(_arm(), [_metrics(clause_2_correlation=correlation)])
    assert summary["clause_2_seeds_cleared"] == int(correlation > rg.CLAUSE_2_THRESHOLD)
    assert summary["mean_clause_2_correlation"] == pytest.approx(correlation)


def test_pooled_precision_sums_reports_rather_than_averaging_rates() -> None:
    summary = rg.summarize_arm(
        _arm(),
        [
            _metrics(reports=10, correct_reports=1),
            _metrics(reports=90, correct_reports=45),
        ],
    )
    assert summary["reports"] == 100
    assert summary["correct_reports"] == 46
    assert summary["precision"] == pytest.approx(0.46)


def test_arm_without_reports_is_unscorable_not_zero() -> None:
    summary = rg.summarize_arm(_arm(), [_metrics(reports=0, correct_reports=0)])
    assert summary["precision"] is None


def _results(
    *,
    slope: float,
    correlation: float,
    precision_reports: tuple[int, int] = (100, 60),
    static_prior: float = 0.35,
) -> dict[str, Any]:
    arms = rg.gate_arms((0.0, 1.0))
    reports, correct = precision_reports
    per_seed = {
        arm.name: [
            _metrics(
                seed=seed,
                reports=reports,
                correct_reports=correct,
                clause_1_slope=slope,
                clause_2_correlation=correlation,
            )
            for seed in (1, 2, 3, 4)
        ]
        for arm in arms
    }
    block = rg.assemble_block(arms, (1, 2, 3, 4), per_seed)
    results: dict[str, Any] = {
        "spec": DesignedReporterSpec(levers=PayoffLevers(enabled=True)).as_dict(),
        "measurement_path": "agent-only test path",
        "nulls": {
            "static_prior_baseline": static_prior,
            "chance_baseline": 0.0025,
            "inferability_precision": 0.8,
            "candidate_locations": 400,
            "event_steps": 100,
            "measured_steps": 200,
        },
        "arm_definitions": [arm.as_dict() for arm in arms],
        "blocks": {"primary": block},
    }
    results["verdicts"] = rg.clause_verdicts(results)
    return results


def test_clause_1_needs_both_a_rise_and_a_rate_above_the_null() -> None:
    above_and_rising = _results(slope=0.003, correlation=0.0, precision_reports=(100, 60))
    below_but_rising = _results(slope=0.003, correlation=0.0, precision_reports=(100, 10))
    above_but_falling = _results(slope=-0.003, correlation=0.0, precision_reports=(100, 60))
    verdicts = [
        result["verdicts"]["arms"]["evolved_w1"]
        for result in (above_and_rising, below_but_rising, above_but_falling)
    ]
    assert [verdict["clause_1_met"] for verdict in verdicts] == [True, False, False]
    assert verdicts[0]["clause_1_precision_above_static_prior"] is True
    assert verdicts[1]["clause_1_precision_above_static_prior"] is False
    assert verdicts[2]["clause_1_seeds_rising"] == 0


def test_clause_2_needs_seeds_above_the_threshold() -> None:
    weak = _results(slope=0.0, correlation=0.1)["verdicts"]["arms"]["evolved_w1"]
    strong = _results(slope=0.0, correlation=0.5)["verdicts"]["arms"]["evolved_w1"]
    assert weak["clause_2_met"] is False
    assert weak["clause_2_seeds_cleared"] == 0
    assert strong["clause_2_met"] is True
    assert strong["clause_2_seeds_cleared"] == 4


def test_verdicts_cover_only_the_evolved_treatment_arms() -> None:
    verdicts = _results(slope=0.0, correlation=0.0)["verdicts"]
    assert set(verdicts["arms"]) == {"evolved_w1"}


def test_markdown_reports_every_arm_seed_and_verdict() -> None:
    results = _results(slope=0.002, correlation=0.3)
    report = rg.markdown_report(results, "uv run --no-sync --no-build python scripts/x.py")
    assert "scripts/x.py" in report
    assert "reproduction_correctness_weight` is the only quantity" in report
    for arm in ("evolved_w0", "evolved_w1", "designed_w1"):
        assert f"`{arm}`" in report
    for seed in (1, 2, 3, 4):
        assert f"| {seed} |" in report
    assert "**met**" in report
    assert "35.00%" in report


def test_markdown_marks_an_unscorable_arm_instead_of_printing_zero() -> None:
    results = _results(slope=0.0, correlation=0.0, precision_reports=(0, 0))
    report = rg.markdown_report(results, "cmd")
    assert "| Correct-report precision | — | — | — |" in report
    assert "| Correct-report precision | 0.00%" not in report


def _blank_run() -> SeedRun:
    """A run in which nothing happened, with no ledger attached."""
    return SeedRun(
        seed=7,
        arm=ORDINARY_ARM,
        reports=0,
        correct_reports=0,
        designed_reports=0,
        designed_correct_reports=0,
        ordinary_reports=0,
        ordinary_correct_reports=0,
        steps_with_fire=0,
        steps_with_correct_report=0,
        reports_per_adult_lifetime=0.0,
        evidence_rates={},
        parent_child_reproductive_correlation=None,
        ecology={"final_population": 0},
    )


def test_a_run_without_adults_yields_neutral_metrics() -> None:
    """Degenerate runs must not crash or invent a clause result."""
    run = dataclasses.replace(_blank_run(), payoff_coupling={"n_adults": 0})
    metrics = rg.seed_metrics(_arm(), run)
    assert metrics["precision"] is None
    assert metrics["clause_1_slope"] == pytest.approx(0.0)
    assert metrics["clause_2_correlation"] == pytest.approx(0.0)
    assert metrics["population_capped_step_share"] == pytest.approx(0.0)


def test_metrics_require_a_ledger() -> None:
    arm = _arm()
    run = _blank_run()
    with pytest.raises(ValueError, match="without a payoff ledger"):
        rg.seed_metrics(arm, run)


@requires_grounded_knobs
@pytest.mark.parametrize("weight", [0.0, 1.0])
def test_measured_seed_metrics_are_bounded_and_finite(weight: float) -> None:
    arm = _arm(name=f"evolved_w{weight:g}", weight=weight)
    metrics = rg.seed_metrics(arm, rg.run_arm_seed(_SMALL_SPEC, arm, 3))
    for key in (
        "silent_adult_share",
        "population_capped_step_share",
        "eligible_share",
    ):
        assert 0.0 <= float(metrics[key]) <= 1.0
    for key in ("clause_2_correlation", "corr_parent_child_precision"):
        assert -1.0 <= float(metrics[key]) <= 1.0
    assert np.isfinite(float(metrics["clause_1_slope"]))
    assert metrics["reports_per_adult_lifetime"] >= 0.0
    assert metrics["correct_reports"] <= metrics["reports"]
    assert metrics["final_population"] <= _SMALL_SPEC.max_population


@requires_grounded_knobs
def test_designed_arm_outreports_the_evolved_arm_under_the_same_levers() -> None:
    arms = rg.gate_arms((1.0,))
    block = rg.run_block(_SMALL_SPEC, arms, (3,))
    evolved = block["arms"]["evolved_w1"]
    designed = block["arms"]["designed_w1"]
    assert designed["reports"] > 0
    assert float(designed["precision"]) > float(evolved["precision"])
    assert designed["mean_reports_per_adult_lifetime"] > evolved["mean_reports_per_adult_lifetime"]
