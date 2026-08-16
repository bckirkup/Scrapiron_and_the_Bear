"""Tests for the designed-reporter measurement of the wildfire exploitable margin.

The measurement claims two things that must be tested separately: that the
hand-designed reporter is a *detector* (its escalations track the published
evidence, graded by the confidence floor, and it stays silent without evidence),
and that the *harness* computes the margin, the pooling and the artifacts
correctly. Golden numbers are avoided; the run-level tests assert bounds,
ordering and sensitivity instead.
"""

from __future__ import annotations

import dataclasses
import inspect
import json

import numpy as np
import pytest
from tattletots.interface.reporter_policy import (
    ReporterMetadata,
    ReporterPolicyContext,
    ReporterStream,
    create_reporter_policy,
)
from tattletots.output_schema import SimulationOutput

from fire_ecology.architectures.ecology_options import engine_supports
from fire_ecology.measurement import designed_reporter as dr
from fire_ecology.reporter_policy import (
    FIRE_REPORTER_POLICY_NAME,
    THERMAL_STREAM_LABEL,
    FireThermalEvidenceReporterPolicy,
)

requires_grounded_knobs = pytest.mark.skipif(
    not engine_supports("grounded_input_fraction"),
    reason="installed TattleTots engine has no grounded_input_fraction knob",
)

#: Small, fire-bearing scenario. The ignition rate is raised well above the
#: measured scenario's default only so that a 30-step test window contains fires;
#: the reported measurement uses the domain's default rate.
_SMALL_SPEC = dr.DesignedReporterSpec(
    steps=30,
    grid_rows=12,
    grid_cols=12,
    n_cameras=2,
    base_ignition_rate=0.01,
    initial_population=6,
    max_population=12,
)


def _thermal_context(
    values: list[float],
    *,
    statuses: list[str] | None = None,
    coordinates: tuple[tuple[float, ...] | None, ...] | None = None,
    declared: bool = True,
    label: str = THERMAL_STREAM_LABEL,
) -> ReporterPolicyContext:
    """A context carrying one synthetic published thermal stream."""
    data = np.asarray(values, dtype=np.float64)
    if statuses is None:
        statuses = ["observed"] * len(values)
    if coordinates is None and declared:
        coordinates = tuple((float(index), 1.0) for index in range(len(values)))
    stream = ReporterStream(
        label=label,
        data=data,
        observation_status=tuple(statuses),
        metadata=ReporterMetadata(coordinates=coordinates),
    )
    return ReporterPolicyContext(
        observation=data,
        signal_vector=data,
        anomaly_score=0.0,
        escalation_threshold=0.0,
        time_step=3,
        location_frame=((0, 0), (9, 9)),
        streams=(stream,),
    )


def _escalations(policy: FireThermalEvidenceReporterPolicy, frames: list[list[float]]) -> int:
    return sum(int(policy.decide(_thermal_context(frame)).escalate) for frame in frames)


_EVIDENCE_FRAMES = [
    [0.0, 0.0, 0.0],
    [0.3, 0.0, 0.0],
    [0.3, 0.5, 0.0],
    [0.7, 0.3, 0.6],
    [0.0, 0.9, 0.0],
]


def test_confidence_floor_grades_escalation_count() -> None:
    """A few floors produce a few different escalation counts, ordered."""
    counts = [
        _escalations(FireThermalEvidenceReporterPolicy(confidence_floor=floor), _EVIDENCE_FRAMES)
        for floor in (0.1, 0.45, 0.65, 0.95)
    ]
    assert counts == sorted(counts, reverse=True)
    assert len(set(counts)) >= 3
    assert counts[0] == 4
    assert counts[-1] == 0


def test_reporter_names_the_strongest_observed_cell() -> None:
    policy = FireThermalEvidenceReporterPolicy()
    coordinates = ((2.0, 3.0), (4.0, 5.0), (6.0, 7.0))
    context = _thermal_context([0.5, 0.9, 0.6], coordinates=coordinates)
    decision = policy.decide(context)
    assert decision.escalate
    assert decision.location == (4, 5)


def test_reporter_stays_silent_without_usable_evidence() -> None:
    """Negative controls: no evidence, no observation, no declarations, wrong stream."""
    policy = FireThermalEvidenceReporterPolicy()
    below_floor = policy.decide(_thermal_context([0.3, 0.2, 0.0]))
    unobserved = policy.decide(
        _thermal_context([0.9, 0.9], statuses=["missing", "missing"]),
    )
    undeclared = policy.decide(_thermal_context([0.9, 0.9], declared=False))
    mismatched = policy.decide(_thermal_context([0.9, 0.9], coordinates=((1.0, 1.0),)))
    other_stream = policy.decide(_thermal_context([0.9], label="weather_observations"))
    assert not below_floor.escalate
    assert not unobserved.escalate
    assert not undeclared.escalate
    assert not mismatched.escalate
    assert not other_stream.escalate


def test_reporter_rejects_non_finite_and_out_of_frame_evidence() -> None:
    policy = FireThermalEvidenceReporterPolicy()
    non_finite = policy.decide(_thermal_context([float("nan"), float("inf")]))
    outside = policy.decide(_thermal_context([0.9], coordinates=((99.0, 99.0),)))
    assert not non_finite.escalate
    assert not outside.escalate


def test_reporter_counters_track_decisions_and_evidence() -> None:
    policy = FireThermalEvidenceReporterPolicy()
    _escalations(policy, _EVIDENCE_FRAMES)
    assert policy.decision_steps == len(_EVIDENCE_FRAMES)
    assert policy.thermal_stream_steps == len(_EVIDENCE_FRAMES)
    assert policy.thermal_evidence_steps < policy.thermal_observed_steps
    assert policy.escalations == policy.thermal_evidence_steps


def test_registered_reporter_policy_is_the_designed_one() -> None:
    policy = create_reporter_policy(FIRE_REPORTER_POLICY_NAME)
    assert isinstance(policy, FireThermalEvidenceReporterPolicy)


def test_designed_reporter_reads_no_ground_truth() -> None:
    """Structural check: the feasible reporter cannot see the fire or its truth."""
    source = inspect.getsource(FireThermalEvidenceReporterPolicy)
    assert "get_active_locations" not in source
    assert "FireGrid" not in source
    assert "active_locations" not in source


def test_oracle_policy_is_ground_truth_and_labelled_diagnostic() -> None:
    oracle = dr.OracleDiagnosticPolicy()
    silent = oracle.decide(_thermal_context([0.9]))
    oracle.active_locations = ((1, 2),)
    informed = oracle.decide(_thermal_context([0.0]))
    assert not silent.escalate
    assert informed.location == (1, 2)
    assert dr.ORACLE_ARM not in dr.FEASIBLE_ARMS


def test_designed_seed_count_is_graded_by_arm_and_fraction() -> None:
    counts = [
        dr.designed_seed_count(dr.ORDINARY_ARM, 20, 0.15),
        dr.designed_seed_count(dr.INVASION_ARM, 20, 0.05),
        dr.designed_seed_count(dr.INVASION_ARM, 20, 0.15),
        dr.designed_seed_count(dr.INVASION_ARM, 20, 0.5),
        dr.designed_seed_count(dr.ALL_DESIGNED_ARM, 20, 0.15),
    ]
    assert counts == sorted(counts)
    assert counts[0] == 0
    assert counts[-1] == 20
    assert len(set(counts)) == 5


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown policy arm"):
        dr.policy_name_for_arm("not_an_arm")


@requires_grounded_knobs
def test_seeded_policies_match_the_arm() -> None:
    for arm, expected in (
        (dr.ORDINARY_ARM, 0),
        (dr.INVASION_ARM, 1),
        (dr.ALL_DESIGNED_ARM, _SMALL_SPEC.initial_population),
    ):
        _, world = dr.build_world(_SMALL_SPEC, seed=11, arm=arm)
        designed = sum(
            1
            for agent in world.agents.values()
            if agent.genome.reporter_policy == FIRE_REPORTER_POLICY_NAME
        )
        assert designed == expected


def _synthetic_results(
    *,
    designed_precision: float,
    ordinary_precision: float,
    designed_reports: int = 100,
    static_prior: float = 0.4,
) -> dict[str, object]:
    def arm(designed: float, ordinary: float, reports: int) -> dict[str, object]:
        return {
            "designed_precision": designed,
            "ordinary_precision": ordinary,
            "precision": max(designed, ordinary),
            "designed_reports": reports,
            "ordinary_reports": 100,
            "reports": reports + 100,
        }

    return {
        "nulls": {
            "static_prior_baseline": static_prior,
            "chance_baseline": 0.01,
            "inferability_precision": 0.8,
            "decoder_precision": 0.7,
        },
        "arms": {
            dr.ORDINARY_ARM: arm(0.0, ordinary_precision, 0),
            dr.ALL_DESIGNED_ARM: arm(designed_precision, 0.0, designed_reports),
            dr.ORACLE_ARM: arm(1.0, 0.0, 500),
        },
    }


@pytest.mark.parametrize("designed_precision", [0.2, 0.45, 0.75, 0.95])
def test_margin_tracks_precision_and_null(designed_precision: float) -> None:
    results = _synthetic_results(designed_precision=designed_precision, ordinary_precision=0.1)
    margin = dr.exploitable_margin(results)
    assert margin["exploitable_margin"] == pytest.approx(designed_precision - 0.4)
    assert margin["margin_is_positive"] is (designed_precision > 0.4)
    assert margin["exploitable_margin_pp"] == pytest.approx(100.0 * (designed_precision - 0.4))


def test_margin_excludes_the_oracle_arm() -> None:
    results = _synthetic_results(designed_precision=0.5, ordinary_precision=0.3)
    margin = dr.exploitable_margin(results)
    assert margin["best_feasible_arm"] == dr.ALL_DESIGNED_ARM
    assert margin["best_feasible_precision"] == pytest.approx(0.5)
    assert margin["oracle_precision"] == pytest.approx(1.0)


def test_arm_without_reports_is_unscorable_not_zero() -> None:
    results = _synthetic_results(designed_precision=0.9, ordinary_precision=0.3, designed_reports=0)
    margin = dr.exploitable_margin(results)
    assert dr.ALL_DESIGNED_ARM in margin["unscorable_arms"]
    assert margin["best_feasible_arm"] == dr.ORDINARY_ARM
    assert margin["best_feasible_precision"] == pytest.approx(0.3)


def test_ordinary_above_null_flag_follows_the_null() -> None:
    below = dr.exploitable_margin(
        _synthetic_results(designed_precision=0.9, ordinary_precision=0.3, static_prior=0.4)
    )
    above = dr.exploitable_margin(
        _synthetic_results(designed_precision=0.9, ordinary_precision=0.5, static_prior=0.4)
    )
    assert below["ordinary_above_static_prior"] is False
    assert above["ordinary_above_static_prior"] is True


@requires_grounded_knobs
def test_designed_reports_grade_with_the_seeded_minority() -> None:
    """Graded sensitivity: a few invasion fractions give a few report volumes."""
    counts = [
        dr.run_seed(
            dataclasses.replace(_SMALL_SPEC, invasion_fraction=fraction),
            3,
            dr.INVASION_ARM,
        ).designed_reports
        for fraction in (0.1, 0.34, 0.67)
    ]
    assert counts == sorted(counts)
    assert len(set(counts)) == 3
    assert counts[0] > 0


@requires_grounded_knobs
def test_without_ignitions_there_is_nothing_to_report() -> None:
    """Negative control: no fire in the window means no event steps and no reports."""
    quiet = dataclasses.replace(_SMALL_SPEC, base_ignition_rate=0.0)
    nulls = dr.instrument_nulls(quiet, seed=5)
    run = dr.run_seed(quiet, 5, dr.ALL_DESIGNED_ARM)
    assert nulls["event_steps"] == 0
    assert run.steps_with_fire == 0
    assert run.designed_correct_reports == 0


@requires_grounded_knobs
def test_nulls_are_bounded_and_finite() -> None:
    nulls = dr.instrument_nulls(_SMALL_SPEC, seed=5)
    for key in (
        "static_prior_baseline",
        "chance_baseline",
        "inferability_precision",
        "decoder_precision",
    ):
        value = float(nulls[key])
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert nulls["chance_baseline"] < nulls["static_prior_baseline"]
    assert nulls["candidate_locations"] == _SMALL_SPEC.grid_rows * _SMALL_SPEC.grid_cols


@pytest.fixture(scope="module")
def small_results() -> dict[str, object]:
    """One short experiment over two seeds, all four arms."""
    return dr.run_experiment(spec=_SMALL_SPEC, seeds=(3, 4))


@requires_grounded_knobs
def test_run_metrics_are_bounded_and_finite(small_results: dict[str, object]) -> None:
    arms = small_results["arms"]
    assert isinstance(arms, dict)
    for summary in arms.values():
        for key in ("precision", "designed_precision", "ordinary_precision"):
            assert 0.0 <= float(summary[key]) <= 1.0
        ecology = summary["ecology"]
        for key in (
            "mean_attention_solvent_fraction",
            "mean_grounded_yield_share",
            "mean_effective_grounded_yield_share",
        ):
            assert 0.0 <= float(ecology[key]) <= 1.0
        assert np.isfinite(float(ecology["mean_per_capita_attention_capacity"]))
        assert float(summary["mean_reports_per_adult_lifetime"]) >= 0.0
        correlation = ecology["mean_parent_child_reproductive_correlation"]
        if correlation is not None:
            assert -1.0 <= float(correlation) <= 1.0


@requires_grounded_knobs
def test_evidence_rates_are_nested_fractions(small_results: dict[str, object]) -> None:
    runs = small_results["runs"]
    assert isinstance(runs, dict)
    for run in runs[dr.ALL_DESIGNED_ARM]:
        rates = run.evidence_rates
        assert rates["adult_designed_steps"] > 0
        assert 0.0 <= rates["thermal_evidence_rate"] <= rates["thermal_observed_rate"]
        assert rates["thermal_observed_rate"] <= rates["thermal_stream_rate"]
        assert rates["thermal_stream_rate"] <= 1.0


@requires_grounded_knobs
def test_ordinary_arm_reports_nothing_designed(small_results: dict[str, object]) -> None:
    """Negative control: without a seeded policy there are no designed reports."""
    arms = small_results["arms"]
    assert isinstance(arms, dict)
    assert arms[dr.ORDINARY_ARM]["designed_reports"] == 0
    assert arms[dr.ORDINARY_ARM]["adult_designed_steps"] == 0
    assert arms[dr.ALL_DESIGNED_ARM]["ordinary_reports"] == 0


@requires_grounded_knobs
def test_designed_arms_report_and_beat_the_evolved_arm(small_results: dict[str, object]) -> None:
    arms = small_results["arms"]
    assert isinstance(arms, dict)
    designed = arms[dr.ALL_DESIGNED_ARM]
    invasion = arms[dr.INVASION_ARM]
    assert designed["designed_reports"] > 0
    assert invasion["designed_reports"] > 0
    assert float(designed["designed_precision"]) > float(
        arms[dr.ORDINARY_ARM]["ordinary_precision"]
    )
    assert float(invasion["designed_precision"]) > float(invasion["ordinary_precision"])


@requires_grounded_knobs
def test_invasion_seeds_fewer_designed_reports_than_all_designed(
    small_results: dict[str, object],
) -> None:
    arms = small_results["arms"]
    assert isinstance(arms, dict)
    designed = int(arms[dr.ALL_DESIGNED_ARM]["designed_reports"])
    invasion = int(arms[dr.INVASION_ARM]["designed_reports"])
    assert invasion < designed


@requires_grounded_knobs
def test_pooled_series_sums_counts_and_averages_rates(small_results: dict[str, object]) -> None:
    runs = small_results["runs"]
    assert isinstance(runs, dict)
    arm_runs = runs[dr.ALL_DESIGNED_ARM]
    pooled = dr.pooled_time_series(arm_runs)
    expected_reports = sum(sum(run.time_series["reports_issued"]) for run in arm_runs)
    assert sum(pooled["reports_issued"]) == expected_reports
    assert max(pooled["grounded_yield_share"]) <= 1.0
    assert len(pooled["population"]) == _SMALL_SPEC.steps


@requires_grounded_knobs
def test_simulation_output_round_trips(small_results: dict[str, object]) -> None:
    output = dr.simulation_output(small_results, dr.ALL_DESIGNED_ARM)
    restored = SimulationOutput.model_validate_json(output.model_dump_json())
    assert restored.ecology_metrics.designed_precision == pytest.approx(
        output.ecology_metrics.designed_precision
    )
    assert restored.domain_metrics["policy_arm"] == dr.ALL_DESIGNED_ARM
    assert restored.run_summary.domain == "fire_ecology"


@requires_grounded_knobs
def test_results_json_is_serializable(small_results: dict[str, object]) -> None:
    payload = json.loads(json.dumps(dr.results_json(small_results), sort_keys=True))
    assert payload["margin"]["static_prior_null"] == pytest.approx(
        small_results["margin"]["static_prior_null"]
    )
    assert len(payload["per_seed"][dr.ALL_DESIGNED_ARM]) == 2


@requires_grounded_knobs
def test_markdown_reports_the_margin_and_the_oracle_caveat(
    small_results: dict[str, object],
) -> None:
    report = dr.markdown_report(small_results)
    margin = small_results["margin"]
    assert f"{margin['exploitable_margin_pp']:+.1f} pp" in report
    assert "oracle" in report.lower()
    assert "static-prior null" in report.lower()
    assert dr.ALL_DESIGNED_ARM in report


@pytest.mark.smoke
@requires_grounded_knobs
def test_smoke_designed_arm_has_a_positive_margin_over_the_null() -> None:
    """End-to-end: the designed arm is scorable and measured against the null."""
    results = dr.run_experiment(
        spec=_SMALL_SPEC,
        seeds=(7,),
        arms=(dr.ORDINARY_ARM, dr.ALL_DESIGNED_ARM),
    )
    margin = results["margin"]
    assert margin["best_feasible_arm"] is not None
    assert np.isfinite(float(margin["exploitable_margin"]))
    assert -1.0 <= float(margin["exploitable_margin"]) <= 1.0
    assert float(margin["best_feasible_precision"]) > float(margin["static_prior_null"])
