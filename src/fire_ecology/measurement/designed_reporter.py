"""Measure the fire domain's exploitable margin with a hand-designed reporter.

The cross-domain sweep that scored this domain at a negative exploitable margin
scored evolved agent arms only. This harness adds the missing arm: a hand-coded,
evidence-only reporter (:mod:`fire_ecology.reporter_policy`) run through the
ordinary economy, next to the evolved arm, an invasion arm, and a diagnostic
oracle ceiling.

Measurement path
----------------
Every number here is agent-only. The harness drives the
:class:`~fire_ecology.adapter.fire_adapter.FireEcologyAdapter` directly and
scores TattleTots reports against the adapter's active fire cells; the OPIR
backstop that :mod:`fire_ecology.architectures.a4_bma` can append to domain
detections is never applied, which is the ``ablate_opir_backstop=True`` path of
:class:`~fire_ecology.architectures.ecology_options.EcologyMeasurementOptions`.
OPIR remains wired as a *sensor*: it still feeds the thermal stream the agents
read and still consumes its random draws. No suppression is dispatched, so the
fire trajectory is identical across arms at a given seed and the arms differ only
in who reports.

Nulls
-----
Precision is always read against the domain's own nulls from
:func:`tattletots.interface.instrument.validate_instrument`: the static-prior
baseline (a well-placed constant guess), the uniform chance baseline, the
inferability precision of the published evidence, and the decoder precision.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from tattletots.engine.config import GenePoolConfig, SimulationConfig
from tattletots.engine.world import World
from tattletots.interface.instrument import validate_instrument
from tattletots.interface.reporter_policy import (
    ReporterDecision,
    ReporterPolicyContext,
    register_reporter_policy,
)
from tattletots.models.agent import Agent, LifecycleStage
from tattletots.models.genome import Genome
from tattletots.models.location import EventLocation
from tattletots.output_schema import (
    EcologyMetrics,
    RunSummary,
    SimulationOutput,
    TimeSeries,
)
from tattletots.telemetry.payoff_ledger import PayoffLedger

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.architectures.ecology_options import EcologyMeasurementOptions
from fire_ecology.measurement.grounded_access import reproductive_correlation
from fire_ecology.reporter_policy import (
    FIRE_REPORTER_POLICY_NAME,
    FireThermalEvidenceReporterPolicy,
)

ORACLE_POLICY_NAME = "fire_oracle_diagnostic_upper_bound"
ORDINARY_ARM = "ordinary"
ALL_DESIGNED_ARM = "all_designed_seed"
INVASION_ARM = "invasion"
ORACLE_ARM = "oracle_upper_bound"
POLICY_ARMS: tuple[str, ...] = (ORDINARY_ARM, ALL_DESIGNED_ARM, INVASION_ARM, ORACLE_ARM)
FEASIBLE_ARMS: tuple[str, ...] = (ORDINARY_ARM, ALL_DESIGNED_ARM, INVASION_ARM)
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(42, 62))
MEASUREMENT_PATH = (
    "agent-only: reports are scored against the adapter's active fire cells and the OPIR "
    "backstop is never appended to detections (the ablate_opir_backstop=True path); OPIR "
    "still feeds the thermal stream the agents read"
)
_SERIES_SUM_KEYS = frozenset(
    {
        "population",
        "reports_issued",
        "correct_reports",
        "false_alarms",
        "missed_events",
        "responses_dispatched",
        "responses_judged_necessary",
        "responses_judged_unnecessary",
        "n_attention_solvent_agents",
        "n_attention_eligible_agents",
        "births",
        "deaths",
        "n_compression_types",
        "designed_reports",
        "ordinary_reports",
        "designed_correct_reports",
        "ordinary_correct_reports",
    }
)


@dataclass
class OracleDiagnosticPolicy:
    """Diagnostic ceiling only: this policy is handed the ground-truth cells.

    It is not a feasible detector and must never be read as one. It exists to
    show what precision the scoring rule allows when localization is free.
    """

    active_locations: tuple[EventLocation, ...] = ()
    decision_steps: int = 0
    escalations: int = 0

    def decide(self, _context: ReporterPolicyContext) -> ReporterDecision:
        self.decision_steps += 1
        if not self.active_locations:
            return ReporterDecision(escalate=False)
        self.escalations += 1
        return ReporterDecision(escalate=True, location=self.active_locations[0])


register_reporter_policy(ORACLE_POLICY_NAME, OracleDiagnosticPolicy)


@dataclass(frozen=True)
class PayoffLevers:
    """The engine's measured payoff levers, off by default.

    Levers 1-4 (verified-correctness attention income, merit-ordered rationing of
    reproduction at the population cap, false-alarm pricing at reachable precision,
    escalation thresholds calibrated in score units) were measured in TattleTots and
    are switched together; ``reproduction_correctness_weight`` is lever 5, the response
    gate, and is the only quantity meant to differ between a control and a treatment
    arm. Nothing here subsidizes an agent, protects a lineage or floors the population:
    the levers change what income a correct report earns, what a false alarm costs, the
    units a threshold is compared in, and the order in which a binding cap is spent.
    """

    enabled: bool = False
    correct_report_attention_value: float = 8.0
    false_alarm_break_even_precision: float = 0.2
    escalation_threshold_range: tuple[float, float] = (0.05, 0.3)
    reproduction_correctness_weight: float = 0.0

    def engine_kwargs(self) -> dict[str, Any]:
        """``SimulationConfig`` keyword arguments for this lever setting."""
        if not self.enabled:
            return {}
        return {
            "correct_report_attention_value": self.correct_report_attention_value,
            "reproduction_merit_ordering": True,
            "escalation_calibration_in_score_units": True,
            "false_alarm_break_even_precision": self.false_alarm_break_even_precision,
            "reproduction_correctness_weight": self.reproduction_correctness_weight,
        }

    def gene_pool(self) -> GenePoolConfig | None:
        """Gene-pool constraints for this lever setting; ``None`` keeps the engine default.

        ``gene_pool`` is a :class:`~tattletots.engine.world.World` argument rather than a
        ``SimulationConfig`` field, so it has to be passed separately — ``SimulationConfig``
        would silently drop it.
        """
        if not self.enabled:
            return None
        return GenePoolConfig(escalation_threshold_range=self.escalation_threshold_range)

    def as_dict(self) -> dict[str, Any]:
        """Serializable description; empty when the levers are off."""
        if not self.enabled:
            return {}
        return {
            "payoff_levers": {
                "correct_report_attention_value": self.correct_report_attention_value,
                "reproduction_merit_ordering": True,
                "escalation_calibration_in_score_units": True,
                "false_alarm_break_even_precision": self.false_alarm_break_even_precision,
                "escalation_threshold_range": list(self.escalation_threshold_range),
                "reproduction_correctness_weight": self.reproduction_correctness_weight,
            }
        }


@dataclass(frozen=True)
class DesignedReporterSpec:
    """Scenario shared by every arm; only the reporter policies differ."""

    steps: int = 200
    grid_rows: int = 20
    grid_cols: int = 20
    n_cameras: int = 3
    opir_cadence: int = 5
    base_ignition_rate: float = 0.0001
    initial_population: int = 20
    max_population: int = 60
    mutation_rate: float = 0.1
    grounded_input_fraction: float = 0.67
    grounded_attractiveness_multiplier: float = 1.0
    max_input_streams: int = 3
    invasion_fraction: float = 0.15
    levers: PayoffLevers = PayoffLevers()

    def options(self) -> EcologyMeasurementOptions:
        """Measurement options: the agent-only, OPIR-ablated detection path."""
        return EcologyMeasurementOptions(
            ablate_opir_backstop=True,
            grounded_input_fraction=self.grounded_input_fraction,
            grounded_attractiveness_multiplier=self.grounded_attractiveness_multiplier,
            max_input_streams=self.max_input_streams,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "n_cameras": self.n_cameras,
            "opir_cadence": self.opir_cadence,
            "base_ignition_rate": self.base_ignition_rate,
            "initial_population": self.initial_population,
            "max_population": self.max_population,
            "mutation_rate": self.mutation_rate,
            "invasion_fraction": self.invasion_fraction,
            **self.options().as_dict(),
            **self.levers.as_dict(),
        }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def policy_name_for_arm(arm: str) -> str | None:
    """Reporter policy seeded in ``arm``, or ``None`` for the evolved arm."""
    if arm == ORDINARY_ARM:
        return None
    if arm in (ALL_DESIGNED_ARM, INVASION_ARM):
        return FIRE_REPORTER_POLICY_NAME
    if arm == ORACLE_ARM:
        return ORACLE_POLICY_NAME
    raise ValueError(f"unknown policy arm {arm!r}")


def designed_seed_count(arm: str, population: int, invasion_fraction: float) -> int:
    """Number of seeded genomes that carry the arm's reporter policy."""
    if arm == ORDINARY_ARM:
        return 0
    if arm == INVASION_ARM:
        return max(1, int(round(population * invasion_fraction)))
    return population


def _adapter_for(spec: DesignedReporterSpec, seed: int) -> FireEcologyAdapter:
    return FireEcologyAdapter(
        grid_rows=spec.grid_rows,
        grid_cols=spec.grid_cols,
        seed=seed,
        n_cameras=spec.n_cameras,
        opir_cadence=spec.opir_cadence,
        base_ignition_rate=spec.base_ignition_rate,
    )


def build_world(
    spec: DesignedReporterSpec,
    seed: int,
    arm: str,
) -> tuple[FireEcologyAdapter, World]:
    """Build the adapter and a world whose seeded genomes carry the arm's policy."""
    adapter = _adapter_for(spec, seed)
    config = SimulationConfig(
        initial_population=spec.initial_population,
        max_population=spec.max_population,
        max_steps=spec.steps,
        seed=seed,
        mutation_rate=spec.mutation_rate,
        **spec.options().engine_kwargs(),
        **spec.levers.engine_kwargs(),
    )
    world = World(config=config, gene_pool=spec.levers.gene_pool())
    for stream in adapter.get_streams():
        world.add_stream(stream)
    for user in adapter.get_users():
        world.add_user(user)

    genomes = [
        Genome.random_genome(
            world.rng,
            n_streams=max(len(world.streams), 1),
            input_preference_slots=config.input_preference_slots,
            n_users=max(len(world.users), 1),
            gene_pool=world.gene_pool,
        )
        for _ in range(config.initial_population)
    ]
    policy_name = policy_name_for_arm(arm)
    n_designed = designed_seed_count(arm, len(genomes), spec.invasion_fraction)
    for genome in genomes[:n_designed]:
        genome.reporter_policy = policy_name
    world.seed_population(genomes=genomes)
    world.set_location_inference(adapter.infer_report_location)
    world.set_location_frame(adapter.get_location_frame())
    return adapter, world


def _set_oracle_locations(world: World, active_locations: Sequence[EventLocation]) -> None:
    locations = tuple(active_locations)
    for policy in world.reporter_policies.values():
        if isinstance(policy, OracleDiagnosticPolicy):
            policy.active_locations = locations


def _designed_policies(world: World) -> list[FireThermalEvidenceReporterPolicy]:
    return [
        policy
        for policy in world.reporter_policies.values()
        if isinstance(policy, FireThermalEvidenceReporterPolicy)
    ]


def _evidence_rates(world: World) -> dict[str, float]:
    """Evidence arrival on the adult steps where a designed policy was consulted."""
    policies = _designed_policies(world)
    decisions = sum(policy.decision_steps for policy in policies)
    return {
        "adult_designed_steps": float(decisions),
        "thermal_stream_rate": _ratio(sum(p.thermal_stream_steps for p in policies), decisions),
        "thermal_observed_rate": _ratio(sum(p.thermal_observed_steps for p in policies), decisions),
        "thermal_evidence_rate": _ratio(sum(p.thermal_evidence_steps for p in policies), decisions),
        "escalation_rate": _ratio(sum(p.escalations for p in policies), decisions),
    }


def _ever_adult(agent: Agent) -> bool:
    if agent.state.lifecycle == LifecycleStage.ADULT:
        return True
    return agent.state.age >= agent.genome.development_duration


def _reports_per_adult_lifetime(world: World) -> float:
    """Mean reports issued per agent that ever reached adulthood."""
    adults = [agent for agent in world.agents.values() if _ever_adult(agent)]
    return _ratio(sum(agent.state.reports_issued for agent in adults), len(adults))


def _ecology_summary(world: World, series: dict[str, Any]) -> dict[str, Any]:
    summary = world.telemetry.summary()
    per_capita = [
        capacity / population
        for capacity, population in zip(
            series["attention_carrying_capacity"], series["population"], strict=True
        )
        if population > 0
    ]
    return {
        "attention_solvent_fraction": float(summary["attention_solvent_fraction"]),
        "mean_attention_carrying_capacity": float(summary["mean_attention_carrying_capacity"]),
        "mean_per_capita_attention_capacity": float(np.mean(per_capita)) if per_capita else 0.0,
        "grounded_yield_share": float(summary["grounded_yield_share"]),
        "effective_grounded_yield_share": float(summary["effective_grounded_yield_share"]),
        "final_population": int(summary["final_population"]),
        "peak_population": int(summary["peak_population"]),
        "total_births": int(summary["total_births"]),
        "total_deaths": int(summary["total_deaths"]),
        "max_trophic_depth": float(summary["max_trophic_depth"]),
        "extinct": world.living_population == 0,
        "initiation_is_degenerate": bool(summary["initiation_is_degenerate"]),
        "initiation_degeneracy_reasons": [
            str(reason) for reason in summary["initiation_degeneracy_reasons"]
        ],
    }


@dataclass
class SeedRun:
    """One arm at one seed, agent-only and OPIR-ablated."""

    seed: int
    arm: str
    reports: int
    correct_reports: int
    designed_reports: int
    designed_correct_reports: int
    ordinary_reports: int
    ordinary_correct_reports: int
    steps_with_fire: int
    steps_with_correct_report: int
    reports_per_adult_lifetime: float
    evidence_rates: dict[str, float]
    parent_child_reproductive_correlation: float | None
    ecology: dict[str, Any]
    time_series: dict[str, Any] = field(default_factory=dict)
    payoff_coupling: dict[str, Any] | None = None
    """:meth:`PayoffLedger.coupling_summary` output, when a ledger was attached."""

    @property
    def designed_precision(self) -> float:
        return _ratio(self.designed_correct_reports, self.designed_reports)

    @property
    def ordinary_precision(self) -> float:
        return _ratio(self.ordinary_correct_reports, self.ordinary_reports)

    @property
    def precision(self) -> float:
        return _ratio(self.correct_reports, self.reports)

    def as_dict(self, *, with_series: bool = False) -> dict[str, Any]:
        record = asdict(self)
        if not with_series:
            record.pop("time_series")
        if self.payoff_coupling is None:
            record.pop("payoff_coupling")
        record["designed_precision"] = self.designed_precision
        record["ordinary_precision"] = self.ordinary_precision
        record["precision"] = self.precision
        record["agent_only_step_detection_rate"] = _ratio(
            self.steps_with_correct_report, self.steps_with_fire
        )
        return record


def run_seed(
    spec: DesignedReporterSpec,
    seed: int,
    arm: str,
    *,
    with_payoff_ledger: bool = False,
) -> SeedRun:
    """Run one arm at one seed and collect its agent-only measurements.

    With ``with_payoff_ledger`` the engine's :class:`PayoffLedger` observes every step
    and the run carries its coupling summary, which is where the two falsification
    clauses and the reproduction-gate shares are measured. The ledger reads public
    agent state only and consumes no random draws, so a run is identical with and
    without it.
    """
    adapter, world = build_world(spec, seed, arm)
    ledger = PayoffLedger() if with_payoff_ledger else None
    steps_with_fire = 0
    steps_with_correct_report = 0
    for step in range(spec.steps):
        adapter.step(step)
        active = adapter.get_active_locations(step)
        world.set_event_state(active)
        _set_oracle_locations(world, active)
        record = world.step()
        if ledger is not None:
            ledger.observe(world)
        steps_with_fire += int(bool(active))
        steps_with_correct_report += int(record.correct_reports > 0)

    if ledger is not None:
        ledger.finalize(world)
    series: dict[str, Any] = world.telemetry.ecology_time_series()
    parents = {agent.id: list(agent.state.parent_ids) for agent in world.agents.values()}
    return SeedRun(
        seed=seed,
        arm=arm,
        reports=int(sum(series["reports_issued"])),
        correct_reports=int(sum(series["correct_reports"])),
        designed_reports=int(sum(series["designed_reports"])),
        designed_correct_reports=int(sum(series["designed_correct_reports"])),
        ordinary_reports=int(sum(series["ordinary_reports"])),
        ordinary_correct_reports=int(sum(series["ordinary_correct_reports"])),
        steps_with_fire=steps_with_fire,
        steps_with_correct_report=steps_with_correct_report,
        reports_per_adult_lifetime=_reports_per_adult_lifetime(world),
        evidence_rates=_evidence_rates(world),
        parent_child_reproductive_correlation=reproductive_correlation(parents),
        ecology=_ecology_summary(world, series),
        time_series=series,
        payoff_coupling=None if ledger is None else ledger.coupling_summary(),
    )


def instrument_nulls(spec: DesignedReporterSpec, seed: int = 42) -> dict[str, Any]:
    """The domain's own nulls, measured on the same adapter the arms run on."""
    report = validate_instrument(_adapter_for(spec, seed), steps=spec.steps)
    return {
        "validation_seed": seed,
        "measured_steps": report.measured_steps,
        "event_steps": report.event_steps,
        "distinct_event_locations": report.distinct_event_locations,
        "candidate_locations": len(report.candidate_locations),
        "inferability_precision": float(report.inferability_precision),
        "decoder_precision": float(report.decoder_precision),
        "static_prior_baseline": float(report.static_prior_baseline),
        "chance_baseline": float(report.chance_baseline),
        "findings": [str(finding) for finding in report.findings],
    }


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return None
    return float(np.mean(numbers))


def _mean_ecology(runs: Sequence[SeedRun]) -> dict[str, Any]:
    def mean_of(key: str) -> float:
        return float(np.mean([float(run.ecology[key]) for run in runs]))

    correlations = [run.parent_child_reproductive_correlation for run in runs]
    return {
        "mean_attention_solvent_fraction": mean_of("attention_solvent_fraction"),
        "mean_attention_carrying_capacity": mean_of("mean_attention_carrying_capacity"),
        "mean_per_capita_attention_capacity": mean_of("mean_per_capita_attention_capacity"),
        "mean_grounded_yield_share": mean_of("grounded_yield_share"),
        "mean_effective_grounded_yield_share": mean_of("effective_grounded_yield_share"),
        "mean_max_trophic_depth": mean_of("max_trophic_depth"),
        "mean_final_population": mean_of("final_population"),
        "total_births": sum(int(run.ecology["total_births"]) for run in runs),
        "total_deaths": sum(int(run.ecology["total_deaths"]) for run in runs),
        "n_extinct_runs": sum(1 for run in runs if run.ecology["extinct"]),
        "n_degenerate_runs": sum(1 for run in runs if run.ecology["initiation_is_degenerate"]),
        "mean_parent_child_reproductive_correlation": _mean_or_none(correlations),
        "n_runs_with_reproductive_correlation": sum(
            1 for value in correlations if value is not None
        ),
    }


def summarize_arm(arm: str, runs: Sequence[SeedRun]) -> dict[str, Any]:
    """Pool one arm over seeds: counts summed, rates averaged."""
    designed_reports = sum(run.designed_reports for run in runs)
    ordinary_reports = sum(run.ordinary_reports for run in runs)
    designed_correct = sum(run.designed_correct_reports for run in runs)
    ordinary_correct = sum(run.ordinary_correct_reports for run in runs)
    reports = sum(run.reports for run in runs)
    correct = sum(run.correct_reports for run in runs)
    return {
        "arm": arm,
        "n_seeds": len(runs),
        "reports": reports,
        "correct_reports": correct,
        "precision": _ratio(correct, reports),
        "designed_reports": designed_reports,
        "designed_correct_reports": designed_correct,
        "designed_precision": _ratio(designed_correct, designed_reports),
        "ordinary_reports": ordinary_reports,
        "ordinary_correct_reports": ordinary_correct,
        "ordinary_precision": _ratio(ordinary_correct, ordinary_reports),
        "seeds_without_designed_reports": sum(1 for run in runs if run.designed_reports == 0),
        "mean_reports_per_adult_lifetime": float(
            np.mean([run.reports_per_adult_lifetime for run in runs])
        ),
        "mean_agent_only_step_detection_rate": float(
            np.mean([_ratio(run.steps_with_correct_report, run.steps_with_fire) for run in runs])
        ),
        "mean_thermal_evidence_rate": float(
            np.mean([run.evidence_rates["thermal_evidence_rate"] for run in runs])
        ),
        "mean_thermal_observed_rate": float(
            np.mean([run.evidence_rates["thermal_observed_rate"] for run in runs])
        ),
        "adult_designed_steps": sum(
            int(run.evidence_rates["adult_designed_steps"]) for run in runs
        ),
        "ecology": _mean_ecology(runs),
    }


def _arm_reported_precision(arm: str, summary: dict[str, Any]) -> float:
    """The precision that arm's own reporters achieved."""
    if arm == ORDINARY_ARM:
        return float(summary["ordinary_precision"])
    if arm == ORACLE_ARM:
        return float(summary["precision"])
    return float(summary["designed_precision"])


def _scorable(arm: str, summary: dict[str, Any]) -> bool:
    """Whether the arm's own reporters issued any report at all."""
    if arm == ORDINARY_ARM:
        return int(summary["ordinary_reports"]) > 0
    if arm == ORACLE_ARM:
        return int(summary["reports"]) > 0
    return int(summary["designed_reports"]) > 0


def exploitable_margin(results: dict[str, Any]) -> dict[str, Any]:
    """Best reachable precision minus the domain's own static-prior null."""
    nulls = results["nulls"]
    static_prior = float(nulls["static_prior_baseline"])
    summaries = results["arms"]
    scorable = [
        (arm, _arm_reported_precision(arm, summaries[arm]))
        for arm in FEASIBLE_ARMS
        if arm in summaries and _scorable(arm, summaries[arm])
    ]
    best_arm, best_precision = max(scorable, key=lambda item: item[1]) if scorable else (None, 0.0)
    ordinary = summaries.get(ORDINARY_ARM)
    oracle = summaries.get(ORACLE_ARM)
    ordinary_precision = float(ordinary["ordinary_precision"]) if ordinary else None
    return {
        "static_prior_null": static_prior,
        "chance_null": float(nulls["chance_baseline"]),
        "inferability_precision": float(nulls["inferability_precision"]),
        "decoder_precision": float(nulls["decoder_precision"]),
        "best_feasible_arm": best_arm,
        "best_feasible_precision": best_precision,
        "exploitable_margin": best_precision - static_prior,
        "exploitable_margin_pp": 100.0 * (best_precision - static_prior),
        "margin_is_positive": best_precision > static_prior,
        "ordinary_precision": ordinary_precision,
        "ordinary_above_static_prior": (
            ordinary_precision is not None and ordinary_precision > static_prior
        ),
        "oracle_precision": float(oracle["precision"]) if oracle else None,
        "unscorable_arms": [
            arm for arm in FEASIBLE_ARMS if arm in summaries and not _scorable(arm, summaries[arm])
        ],
    }


def assemble_results(
    spec: DesignedReporterSpec,
    seeds: Sequence[int],
    runs: dict[str, list[SeedRun]],
) -> dict[str, Any]:
    """Pool completed runs, measure the nulls, and derive the margin."""
    results: dict[str, Any] = {
        "spec": spec.as_dict(),
        "seeds": list(seeds),
        "measurement_path": MEASUREMENT_PATH,
        "nulls": instrument_nulls(spec),
        "runs": runs,
        "arms": {arm: summarize_arm(arm, arm_runs) for arm, arm_runs in runs.items()},
    }
    results["margin"] = exploitable_margin(results)
    return results


def run_experiment(
    spec: DesignedReporterSpec | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    arms: Sequence[str] = POLICY_ARMS,
) -> dict[str, Any]:
    """Run every arm over every seed and pool the results."""
    spec = spec or DesignedReporterSpec()
    runs = {arm: [run_seed(spec, seed, arm) for seed in seeds] for arm in arms}
    return assemble_results(spec, seeds, runs)


def pooled_time_series(runs: Sequence[SeedRun]) -> dict[str, list[Any]]:
    """Pool per-seed series: counts are summed, rates averaged, step by step."""
    length = min(len(run.time_series["population"]) for run in runs)
    pooled: dict[str, list[Any]] = {}
    for key in runs[0].time_series:
        stacked = np.asarray(
            [run.time_series[key][:length] for run in runs],
            dtype=np.float64,
        )
        if key in _SERIES_SUM_KEYS:
            pooled[key] = [int(value) for value in stacked.sum(axis=0)]
        else:
            pooled[key] = [float(value) for value in stacked.mean(axis=0)]
    return pooled


def _pooled_ecology_metrics(
    arm: str,
    summary: dict[str, Any],
    nulls: dict[str, Any],
) -> EcologyMetrics:
    ecology = summary["ecology"]
    return EcologyMetrics(
        final_population=int(round(float(ecology["mean_final_population"]))),
        total_births=int(ecology["total_births"]),
        total_deaths=int(ecology["total_deaths"]),
        total_reports=int(summary["reports"]),
        precision=float(summary["precision"]),
        chance_precision=float(nulls["chance_baseline"]),
        static_prior_precision=float(nulls["static_prior_baseline"]),
        location_support_size=int(nulls["candidate_locations"]),
        grounded_yield_share=float(ecology["mean_grounded_yield_share"]),
        effective_grounded_yield_share=float(ecology["mean_effective_grounded_yield_share"]),
        attention_solvent_fraction=float(ecology["mean_attention_solvent_fraction"]),
        mean_attention_carrying_capacity=float(ecology["mean_attention_carrying_capacity"]),
        max_trophic_depth=float(ecology["mean_max_trophic_depth"]),
        designed_precision=_arm_reported_precision(arm, summary),
        ordinary_precision=float(summary["ordinary_precision"]),
    )


def simulation_output(results: dict[str, Any], arm: str) -> SimulationOutput:
    """Seed-pooled :class:`SimulationOutput` for one policy arm."""
    runs: list[SeedRun] = list(results["runs"][arm])
    summary = results["arms"][arm]
    series = pooled_time_series(runs)
    spec = results["spec"]
    return SimulationOutput(
        run_summary=RunSummary(
            domain="fire_ecology",
            steps_completed=int(spec["steps"]),
            seed=None,
        ),
        simulation_config={**spec, "seeds": list(results["seeds"]), "policy_arm": arm},
        domain_config={
            "architecture": "designed reporter measurement (agent-only, OPIR-ablated)",
            "policy_arm": arm,
            "reporter_policy": policy_name_for_arm(arm),
            "seeds": list(results["seeds"]),
        },
        ecology_metrics=_pooled_ecology_metrics(arm, summary, results["nulls"]),
        domain_metrics={
            "policy_arm": arm,
            "measurement_path": results["measurement_path"],
            "nulls": results["nulls"],
            "arm_summary": summary,
            "margin": results["margin"],
            "per_seed": [run.as_dict() for run in runs],
            "pooling": "counts summed across seeds; rates averaged across seeds",
        },
        time_series=TimeSeries(**series),
    )


def _arm_label(arm: str) -> str:
    if arm == ALL_DESIGNED_ARM:
        return "all-designed seed"
    if arm == ORACLE_ARM:
        return "oracle diagnostic upper bound"
    return arm


def _header_lines(results: dict[str, Any]) -> list[str]:
    spec = results["spec"]
    nulls = results["nulls"]
    margin = results["margin"]
    return [
        "# Wildfire designed-reporter measurement",
        "",
        "## The question",
        "",
        "What is the exploitable margin of the wildfire domain — the best reachable",
        "report precision minus the domain's own static-prior null — and is it positive?",
        "",
        f"- Static-prior null: **{nulls['static_prior_baseline']:.2%}**",
        f"- Best reachable precision: **{margin['best_feasible_precision']:.2%}** "
        f"(arm `{margin['best_feasible_arm']}`)",
        f"- **Exploitable margin: {margin['exploitable_margin_pp']:+.1f} pp** "
        f"({'positive' if margin['margin_is_positive'] else 'not positive'})",
        f"- Evolved (`ordinary`) arm precision: **{_format_optional(margin['ordinary_precision'])}**"
        f" — {'above' if margin['ordinary_above_static_prior'] else 'not above'} the null",
        "",
        "The oracle arm is excluded from the margin: it is handed the ground-truth cells",
        "and is a diagnostic ceiling on the scoring rule, not a reachable detector.",
        f"Its pooled precision is {_format_optional(margin['oracle_precision'])}.",
        "",
        "## Measurement path",
        "",
        results["measurement_path"] + ".",
        "",
        "No suppression is dispatched and no subsidy, grace period, juvenile discount or",
        "population floor is applied; the arms differ only in which reporter policy the",
        "seeded genomes carry.",
        "",
        f"- Seeds: `{', '.join(str(seed) for seed in results['seeds'])}`",
        f"- Steps per run: `{spec['steps']}`, grid `{spec['grid_rows']}x{spec['grid_cols']}`, "
        f"cameras `{spec['n_cameras']}`, OPIR cadence `{spec['opir_cadence']}`, "
        f"base ignition rate `{spec['base_ignition_rate']}`",
        f"- Initial population `{spec['initial_population']}`, "
        f"max population `{spec['max_population']}`, mutation rate `{spec['mutation_rate']}`",
        f"- Grounded raw-stream access: `grounded_input_fraction="
        f"{spec['grounded_input_fraction']}`, `grounded_attractiveness_multiplier="
        f"{spec['grounded_attractiveness_multiplier']}`, "
        f"`max_input_streams={spec['max_input_streams']}`",
        f"- Invasion arm seeds the designed policy in {spec['invasion_fraction']:.0%} of genomes",
        "",
        "## Nulls (`validate_instrument`, same adapter and step count)",
        "",
        "| Null / reference | Value |",
        "|---|---:|",
        f"| Static-prior baseline (best constant guess) | {nulls['static_prior_baseline']:.2%} |",
        f"| Uniform chance baseline | {nulls['chance_baseline']:.2%} |",
        f"| Inferability precision of published evidence | {nulls['inferability_precision']:.2%} |",
        f"| Decoder precision | {nulls['decoder_precision']:.2%} |",
        f"| Candidate locations | {nulls['candidate_locations']} |",
        f"| Event steps in the measured window | {nulls['event_steps']} / "
        f"{nulls['measured_steps']} |",
    ]


def _format_optional(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def _precision_table(results: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Arms",
        "",
        "| Policy arm | Reporter precision | Reports scored | Designed precision | "
        "Ordinary precision | Reports per adult lifetime | Seeds without designed reports |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in POLICY_ARMS:
        summary = results["arms"].get(arm)
        if summary is None:
            continue
        scored = (
            summary["reports"]
            if arm == ORACLE_ARM
            else (
                summary["ordinary_reports"] if arm == ORDINARY_ARM else summary["designed_reports"]
            )
        )
        lines.append(
            f"| {_arm_label(arm)} | {_arm_reported_precision(arm, summary):.2%} | {scored} | "
            f"{summary['designed_precision']:.2%} | {summary['ordinary_precision']:.2%} | "
            f"{summary['mean_reports_per_adult_lifetime']:.2f} | "
            f"{summary['seeds_without_designed_reports']} |"
        )
    return lines


def _ecology_table(results: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "| Policy arm | Evidence on adult designed steps | Thermal coverage | "
        "Attention solvency | Per-capita attention capacity | Grounded yield share | "
        "Parent–child reproductive r | Runs with r | Mean final population | Extinct runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in POLICY_ARMS:
        summary = results["arms"].get(arm)
        if summary is None:
            continue
        ecology = summary["ecology"]
        correlation = ecology["mean_parent_child_reproductive_correlation"]
        rendered = "—" if correlation is None else f"{correlation:+.3f}"
        lines.append(
            f"| {_arm_label(arm)} | {summary['mean_thermal_evidence_rate']:.2%} | "
            f"{summary['mean_thermal_observed_rate']:.2%} | "
            f"{ecology['mean_attention_solvent_fraction']:.2%} | "
            f"{ecology['mean_per_capita_attention_capacity']:.3f} | "
            f"{ecology['mean_grounded_yield_share']:.2%} | {rendered} | "
            f"{ecology['n_runs_with_reproductive_correlation']} | "
            f"{ecology['mean_final_population']:.1f} | {ecology['n_extinct_runs']} |"
        )
    return lines


def _per_seed_table(results: dict[str, Any], arm: str) -> list[str]:
    runs: list[SeedRun] = list(results["runs"].get(arm, []))
    lines = [
        "",
        f"Per-seed designed reports in the `{arm}` arm:",
        "",
        "| Seed | Designed reports | Designed correct | Designed precision | "
        "Ordinary reports | Ordinary precision |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run.seed} | {run.designed_reports} | {run.designed_correct_reports} | "
            f"{run.designed_precision:.2%} | {run.ordinary_reports} | "
            f"{run.ordinary_precision:.2%} |"
        )
    return lines


def _interpretation(results: dict[str, Any]) -> list[str]:
    margin = results["margin"]
    designed = results["arms"].get(ALL_DESIGNED_ARM, {})
    verdict = (
        "positive: a hand-designed reporter reading only the domain's published "
        "evidence beats the best constant guess"
        if margin["margin_is_positive"]
        else "not positive: no feasible arm beat the best constant guess"
    )
    lines = [
        "",
        "## Interpretation",
        "",
        f"The exploitable margin is {margin['exploitable_margin_pp']:+.1f} pp, i.e. {verdict}.",
        "",
        "The designed reporter reads only the published `thermal_detection` values, their",
        "per-feature observation status and the declared coordinates of the observed",
        "features, and requires a published detection confidence above the value the OPIR",
        "sensor stamps on a false positive. It never reads the fire grid or the",
        "ground-truth cells.",
        "",
        f"Evidence arrives on {designed.get('mean_thermal_evidence_rate', 0.0):.2%} of the adult"
        " steps on which a designed policy was consulted in the all-designed arm"
        f" ({designed.get('adult_designed_steps', 0)} such steps pooled); the thermal stream is"
        f" observed on {designed.get('mean_thermal_observed_rate', 0.0):.2%} of them.",
        "",
        "The designed precision can exceed the instrument's inferability precision: the",
        "instrument reads the strongest published feature on every event step, while the",
        "designed reporter declines to report when no published feature clears the",
        "false-positive confidence. It buys precision by staying silent, which the report",
        "count and the reports-per-adult-lifetime column show the price of.",
        "",
        "The parent–child reproductive correlation is the Pearson correlation between a",
        "parent's offspring count and its child's offspring count over all parent-child",
        "pairs in a run, averaged over the runs where both series vary. It is reported as",
        "measured; this measurement does not claim either falsification clause cleared.",
        "",
        "Precision is a pooled count ratio: correct reports over reports issued by that",
        "arm's own reporters. An arm whose reporters issued no report is not scorable and",
        "is listed as such rather than shown as 0%.",
    ]
    if margin["unscorable_arms"]:
        lines.extend(
            [
                "",
                "Unscorable arms (no reports from their own reporters): "
                + ", ".join(f"`{arm}`" for arm in margin["unscorable_arms"]),
            ]
        )
    return lines


_RELATED_MEASUREMENTS: tuple[str, ...] = (
    "",
    "## Related measurements",
    "",
    "- [Response gate (lever 5)](response_gate_measurement.md) — whether keying",
    "  reproductive merit on verified correctness moves either falsification clause on",
    "  this instrument, measured on the same agent-only OPIR-ablated path.",
)


def markdown_report(results: dict[str, Any]) -> str:
    """Render the documentation artifact for one experiment."""
    lines = _header_lines(results)
    lines.extend(_precision_table(results))
    lines.extend(_ecology_table(results))
    lines.extend(_per_seed_table(results, ALL_DESIGNED_ARM))
    lines.extend(_per_seed_table(results, INVASION_ARM))
    lines.extend(_interpretation(results))
    lines.extend(_RELATED_MEASUREMENTS)
    return "\n".join(lines) + "\n"


def results_json(results: dict[str, Any]) -> dict[str, Any]:
    """Serializable results without the per-seed time series."""
    return {
        "spec": results["spec"],
        "seeds": results["seeds"],
        "measurement_path": results["measurement_path"],
        "nulls": results["nulls"],
        "arms": results["arms"],
        "margin": results["margin"],
        "per_seed": {arm: [run.as_dict() for run in runs] for arm, runs in results["runs"].items()},
    }
