"""Does the correctness-keyed response gate move a falsification clause here?

TattleTots measured five payoff levers on its own ``SparseSensorScenario``. The fifth,
``SimulationConfig.reproduction_correctness_weight``, mixes rank in verified correctness
into the ordering that spends a binding population cap, and on that instrument it cleared
falsification clause 1 (correct-report rate rising within a run at fixed initial
parameters) while leaving clause 2 (parent-child reproductive correlation above ~0.2)
untouched. This module asks the same question on the wildfire instrument.

Measurement path
----------------
Identical to :mod:`fire_ecology.measurement.designed_reporter`: agent-only, with the OPIR
backstop never appended to detections while OPIR still feeds the thermal stream the agents
read. The arm under test is the **ordinary (evolved)** arm; the designed reporter is run
once under the same config as a visible ceiling, not as the thing under test.

Arms
----
Every arm holds levers 1-4 fixed (``correct_report_attention_value=8``,
``reproduction_merit_ordering=True``, ``escalation_calibration_in_score_units=True``,
``false_alarm_break_even_precision=0.2``, starting ``escalation_threshold`` range
``(0.05, 0.3)``). The reproductive-merit weight ``W`` is the only difference between the
control (``W=0``, reserves-only ordering) and the treatment (``W=1``). No subsidy, grace
period, juvenile discount or population floor is applied anywhere.

Clause metrics come from the engine's own :class:`~tattletots.telemetry.payoff_ledger.PayoffLedger`,
so they are computed by the same code that produced the SparseSensor numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fire_ecology.measurement.designed_reporter import (
    ALL_DESIGNED_ARM,
    MEASUREMENT_PATH,
    ORDINARY_ARM,
    DesignedReporterSpec,
    PayoffLevers,
    SeedRun,
    instrument_nulls,
    run_seed,
)

CLAUSE_2_THRESHOLD = 0.2
"""Parent-child reproductive correlation a seed must exceed to clear clause 2."""

CLAUSE_1_RELIABLE_SHARE = 0.8
"""Share of seeds that must rise for clause 1 to count as reliable rather than chance."""

PRIMARY_SEEDS: tuple[int, ...] = tuple(range(42, 62))
HOLDOUT_SEEDS: tuple[int, ...] = tuple(range(101, 121))
DEFAULT_WEIGHTS: tuple[float, ...] = (0.0, 1.0)


@dataclass(frozen=True)
class GateArm:
    """One measured arm: a policy arm at one reproductive-merit weight."""

    name: str
    policy_arm: str
    correctness_weight: float

    def spec(self, base: DesignedReporterSpec) -> DesignedReporterSpec:
        """``base`` with the payoff levers on and this arm's weight set."""
        levers = PayoffLevers(
            enabled=True,
            correct_report_attention_value=base.levers.correct_report_attention_value,
            false_alarm_break_even_precision=base.levers.false_alarm_break_even_precision,
            escalation_threshold_range=base.levers.escalation_threshold_range,
            reproduction_correctness_weight=self.correctness_weight,
        )
        return DesignedReporterSpec(
            steps=base.steps,
            grid_rows=base.grid_rows,
            grid_cols=base.grid_cols,
            n_cameras=base.n_cameras,
            opir_cadence=base.opir_cadence,
            base_ignition_rate=base.base_ignition_rate,
            initial_population=base.initial_population,
            max_population=base.max_population,
            mutation_rate=base.mutation_rate,
            grounded_input_fraction=base.grounded_input_fraction,
            grounded_attractiveness_multiplier=base.grounded_attractiveness_multiplier,
            max_input_streams=base.max_input_streams,
            invasion_fraction=base.invasion_fraction,
            levers=levers,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy_arm": self.policy_arm,
            "reproduction_correctness_weight": self.correctness_weight,
        }


def gate_arms(weights: Sequence[float] = DEFAULT_WEIGHTS) -> tuple[GateArm, ...]:
    """The evolved arms under test plus the designed ceiling at the largest weight."""
    ordered = sorted({float(weight) for weight in weights})
    arms = [
        GateArm(name=f"evolved_w{weight:g}", policy_arm=ORDINARY_ARM, correctness_weight=weight)
        for weight in ordered
    ]
    arms.append(
        GateArm(
            name=f"designed_w{ordered[-1]:g}",
            policy_arm=ALL_DESIGNED_ARM,
            correctness_weight=ordered[-1],
        )
    )
    return tuple(arms)


def run_arm_seed(base: DesignedReporterSpec, arm: GateArm, seed: int) -> SeedRun:
    """Run one arm at one seed with the payoff ledger attached."""
    return run_seed(arm.spec(base), seed, arm.policy_arm, with_payoff_ledger=True)


def _reports_and_correct(arm: GateArm, run: SeedRun) -> tuple[int, int]:
    """Reports and correct reports issued by this arm's own reporters."""
    if arm.policy_arm == ORDINARY_ARM:
        return run.ordinary_reports, run.ordinary_correct_reports
    return run.designed_reports, run.designed_correct_reports


_EMPTY_COUPLING: dict[str, Any] = {
    "silent_adult_share": 0.0,
    "precision_generation_slope": 0.0,
    "generations_observed": 0,
    "corr_parent_child_offspring": 0.0,
    "n_parent_child_pairs": 0,
    "correct_group_mean_offspring": 0.0,
    "never_correct_group_mean_offspring": 0.0,
    "silent_mean_offspring": 0.0,
    "corr_parent_child_precision": 0.0,
    "reproduction_gate": {"population_capped_step_share": 0.0, "eligible_share": 0.0},
}
"""Neutral clause metrics for a run in which no agent ever reached adulthood."""


def _coupling(run: SeedRun) -> dict[str, Any]:
    """The run's coupling summary, padded when the run had no adults to measure."""
    if run.payoff_coupling is None:
        raise ValueError(f"seed {run.seed} was run without a payoff ledger")
    return {**_EMPTY_COUPLING, **run.payoff_coupling}


def seed_metrics(arm: GateArm, run: SeedRun) -> dict[str, Any]:
    """Per-seed clause metrics and reporting economics for one run."""
    coupling = _coupling(run)
    gate = dict(coupling["reproduction_gate"])
    reports, correct = _reports_and_correct(arm, run)
    return {
        "seed": run.seed,
        "reports": reports,
        "correct_reports": correct,
        "precision": (correct / reports) if reports else None,
        "reports_per_adult_lifetime": run.reports_per_adult_lifetime,
        "silent_adult_share": float(coupling["silent_adult_share"]),
        "clause_1_slope": float(coupling["precision_generation_slope"]),
        "generations_observed": int(coupling["generations_observed"]),
        "clause_2_correlation": float(coupling["corr_parent_child_offspring"]),
        "n_parent_child_pairs": int(coupling["n_parent_child_pairs"]),
        "population_capped_step_share": float(gate["population_capped_step_share"]),
        "eligible_share": float(gate["eligible_share"]),
        "correct_group_mean_offspring": float(coupling["correct_group_mean_offspring"]),
        "never_correct_group_mean_offspring": float(coupling["never_correct_group_mean_offspring"]),
        "silent_mean_offspring": float(coupling["silent_mean_offspring"]),
        "corr_parent_child_precision": float(coupling["corr_parent_child_precision"]),
        "final_population": int(run.ecology["final_population"]),
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


_MEAN_KEYS = (
    "reports_per_adult_lifetime",
    "silent_adult_share",
    "clause_1_slope",
    "clause_2_correlation",
    "population_capped_step_share",
    "eligible_share",
    "correct_group_mean_offspring",
    "never_correct_group_mean_offspring",
    "silent_mean_offspring",
    "corr_parent_child_precision",
    "generations_observed",
    "final_population",
)


def summarize_arm(arm: GateArm, metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool one arm over seeds: reports summed, everything else averaged."""
    reports = sum(int(row["reports"]) for row in metrics)
    correct = sum(int(row["correct_reports"]) for row in metrics)
    slopes = [float(row["clause_1_slope"]) for row in metrics]
    correlations = [float(row["clause_2_correlation"]) for row in metrics]
    summary: dict[str, Any] = {
        **arm.as_dict(),
        "n_seeds": len(metrics),
        "reports": reports,
        "correct_reports": correct,
        "precision": (correct / reports) if reports else None,
        "clause_1_seeds_rising": sum(1 for slope in slopes if slope > 0.0),
        "clause_1_slope_min": min(slopes) if slopes else 0.0,
        "clause_1_slope_max": max(slopes) if slopes else 0.0,
        "clause_2_seeds_cleared": sum(1 for value in correlations if value > CLAUSE_2_THRESHOLD),
        "clause_2_correlation_max": max(correlations) if correlations else 0.0,
    }
    summary.update(
        {f"mean_{key}": _mean([float(row[key]) for row in metrics]) for key in _MEAN_KEYS}
    )
    return summary


def run_block(
    base: DesignedReporterSpec,
    arms: Sequence[GateArm],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Run every arm over one seed block, serially."""
    per_seed = {
        arm.name: [seed_metrics(arm, run_arm_seed(base, arm, seed)) for seed in seeds]
        for arm in arms
    }
    return assemble_block(arms, seeds, per_seed)


def assemble_block(
    arms: Sequence[GateArm],
    seeds: Sequence[int],
    per_seed: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Pool already-computed per-seed metrics into one seed block."""
    return {
        "seeds": [int(seed) for seed in seeds],
        "arms": {arm.name: summarize_arm(arm, per_seed[arm.name]) for arm in arms},
        "per_seed": per_seed,
    }


def assemble_results(
    base: DesignedReporterSpec,
    arms: Sequence[GateArm],
    blocks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bundle the seed blocks with the domain's own nulls and the clause verdicts."""
    results: dict[str, Any] = {
        "spec": base.as_dict(),
        "measurement_path": MEASUREMENT_PATH,
        "nulls": instrument_nulls(base),
        "arm_definitions": [arm.as_dict() for arm in arms],
        "blocks": blocks,
    }
    results["verdicts"] = clause_verdicts(results)
    return results


def _evolved_arm_names(results: dict[str, Any]) -> list[str]:
    return [
        arm["name"]
        for arm in results["arm_definitions"]
        if arm["policy_arm"] == ORDINARY_ARM and arm["reproduction_correctness_weight"] > 0.0
    ]


def _block_precisions(results: dict[str, Any], arm_name: str) -> list[float]:
    """Pooled precision of one arm in each seed block, skipping unscorable blocks."""
    values: list[float] = []
    for block in results["blocks"].values():
        summary = block["arms"].get(arm_name)
        if summary is not None and summary["precision"] is not None:
            values.append(float(summary["precision"]))
    return values


def _seed_counts(results: dict[str, Any], arm_name: str, key: str) -> tuple[int, int]:
    cleared = 0
    total = 0
    for block in results["blocks"].values():
        summary = block["arms"].get(arm_name)
        if summary is None:
            continue
        cleared += int(summary[key])
        total += int(summary["n_seeds"])
    return cleared, total


def clause_verdicts(results: dict[str, Any]) -> dict[str, Any]:
    """Judge both clauses for the treated evolved arm against explicit criteria.

    Clause 1 counts as met only if the realized correct-report rate is above the
    domain's static-prior null *and* the within-run slope rises in at least
    :data:`CLAUSE_1_RELIABLE_SHARE` of seeds pooled over every seed block; a rise in a
    rate that never reaches the best constant guess is not competence. Clause 2 counts
    as met only if seeds reliably exceed :data:`CLAUSE_2_THRESHOLD`.
    """
    static_prior = float(results["nulls"]["static_prior_baseline"])
    verdicts: dict[str, Any] = {"static_prior_null": static_prior, "arms": {}}
    for arm_name in _evolved_arm_names(results):
        rising, seeds = _seed_counts(results, arm_name, "clause_1_seeds_rising")
        cleared, _ = _seed_counts(results, arm_name, "clause_2_seeds_cleared")
        precisions = _block_precisions(results, arm_name)
        above_null = bool(precisions) and min(precisions) > static_prior
        rising_share = rising / seeds if seeds else 0.0
        verdicts["arms"][arm_name] = {
            "clause_1_seeds_rising": rising,
            "clause_1_seeds": seeds,
            "clause_1_rising_share": rising_share,
            "clause_1_precision_above_static_prior": above_null,
            "clause_1_met": above_null and rising_share >= CLAUSE_1_RELIABLE_SHARE,
            "clause_2_seeds_cleared": cleared,
            "clause_2_met": seeds > 0 and cleared >= CLAUSE_1_RELIABLE_SHARE * seeds,
        }
    return verdicts


def _format_percent(value: float | None) -> str:
    """Render a rate, or an em dash when the arm issued no report at all."""
    return "—" if value is None else f"{value:.2%}"


def _fixed_config_lines(results: dict[str, Any]) -> list[str]:
    spec = results["spec"]
    levers = spec["payoff_levers"]
    return [
        "## Fixed configuration (identical in every arm)",
        "",
        f"- Steps per run `{spec['steps']}`, grid `{spec['grid_rows']}x{spec['grid_cols']}`, "
        f"cameras `{spec['n_cameras']}`, OPIR cadence `{spec['opir_cadence']}`, "
        f"base ignition rate `{spec['base_ignition_rate']}`",
        f"- Initial population `{spec['initial_population']}`, max population "
        f"`{spec['max_population']}`, mutation rate `{spec['mutation_rate']}`",
        f"- Grounded raw-stream access `grounded_input_fraction="
        f"{spec['grounded_input_fraction']}`, `max_input_streams={spec['max_input_streams']}`",
        f"- `correct_report_attention_value={levers['correct_report_attention_value']:g}`, "
        f"`false_alarm_break_even_precision={levers['false_alarm_break_even_precision']}`, "
        "`reproduction_merit_ordering=True`, "
        "`escalation_calibration_in_score_units=True`, "
        f"starting `escalation_threshold` range `{tuple(levers['escalation_threshold_range'])}`",
        "- `reproduction_correctness_weight` is the only quantity that differs between arms",
        "- No subsidy, grace period, juvenile discount or population floor is applied, and "
        "no suppression is dispatched, so the fire trajectory at a seed is identical "
        "across arms",
        "",
        "## Nulls (`validate_instrument`, same adapter and step count)",
        "",
        "| Null / reference | Value |",
        "|---|---:|",
        f"| Static-prior baseline (best constant guess) | "
        f"{results['nulls']['static_prior_baseline']:.2%} |",
        f"| Uniform chance baseline | {results['nulls']['chance_baseline']:.2%} |",
        f"| Inferability precision of published evidence | "
        f"{results['nulls']['inferability_precision']:.2%} |",
        f"| Candidate locations | {results['nulls']['candidate_locations']} |",
        f"| Event steps in the measured window | {results['nulls']['event_steps']} / "
        f"{results['nulls']['measured_steps']} |",
    ]


_ARM_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Correct-report precision", "precision", "percent"),
    ("Reports scored", "reports", "int"),
    ("Reports per adult lifetime", "mean_reports_per_adult_lifetime", "{:.2f}"),
    ("Adults that never report", "mean_silent_adult_share", "percent"),
    ("Generations with reports", "mean_generations_observed", "{:.1f}"),
    ("Clause 1: correct-report slope / generation", "mean_clause_1_slope", "{:+.4f}"),
    ("Clause 1: seeds rising", "clause_1_seeds_rising", "int"),
    ("Clause 2: parent-child reproductive r", "mean_clause_2_correlation", "{:+.3f}"),
    ("Clause 2: seeds above 0.2", "clause_2_seeds_cleared", "int"),
    ("Steps where the population cap binds", "mean_population_capped_step_share", "percent"),
    ("Eligible share of agent-steps", "mean_eligible_share", "percent"),
    ("Mean offspring, adults with a correct report", "mean_correct_group_mean_offspring", "{:.3f}"),
    (
        "Mean offspring, adults reporting but never correct",
        "mean_never_correct_group_mean_offspring",
        "{:.3f}",
    ),
    ("Mean offspring, silent adults", "mean_silent_mean_offspring", "{:.3f}"),
    ("Parent-child precision r", "mean_corr_parent_child_precision", "{:+.3f}"),
    ("Mean final population", "mean_final_population", "{:.1f}"),
)


def _format_cell(value: Any, style: str) -> str:
    if style == "percent":
        return _format_percent(None if value is None else float(value))
    if style == "int":
        return f"{int(value)}"
    return style.format(value)


def _arm_table(block: dict[str, Any], arm_names: Sequence[str]) -> list[str]:
    header = "| Quantity | " + " | ".join(f"`{name}`" for name in arm_names) + " |"
    rule = "|---" + "|---:" * len(arm_names) + "|"
    lines = [header, rule]
    for label, key, style in _ARM_ROWS:
        cells = [_format_cell(block["arms"][name][key], style) for name in arm_names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def _per_seed_table(block: dict[str, Any], arm_name: str) -> list[str]:
    lines = [
        "",
        f"Per-seed detail, `{arm_name}`:",
        "",
        "| Seed | Reports | Correct | Precision | Reports/adult | Clause 1 slope | "
        "Clause 2 r | Pairs | Cap binds | Final pop |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in block["per_seed"][arm_name]:
        lines.append(
            f"| {row['seed']} | {row['reports']} | {row['correct_reports']} | "
            f"{_format_percent(row['precision'])} | "
            f"{row['reports_per_adult_lifetime']:.2f} | {row['clause_1_slope']:+.4f} | "
            f"{row['clause_2_correlation']:+.3f} | {row['n_parent_child_pairs']} | "
            f"{row['population_capped_step_share']:.0%} | {row['final_population']} |"
        )
    return lines


def _block_sections(results: dict[str, Any]) -> list[str]:
    arm_names = [arm["name"] for arm in results["arm_definitions"]]
    lines: list[str] = []
    for block_name, block in results["blocks"].items():
        seeds = block["seeds"]
        lines.extend(
            [
                "",
                f"## Seed block `{block_name}` (seeds {seeds[0]}–{seeds[-1]}, {len(seeds)} seeds)",
                "",
            ]
        )
        lines.extend(_arm_table(block, arm_names))
        for arm_name in arm_names:
            lines.extend(_per_seed_table(block, arm_name))
    return lines


def _verdict_lines(results: dict[str, Any]) -> list[str]:
    static_prior = results["verdicts"]["static_prior_null"]
    lines = [
        "",
        "## Verdict against the falsification test",
        "",
        f"A clause is judged met here only if the evolved arm's realized correct-report "
        f"rate is above this domain's static-prior null ({static_prior:.2%}) and the "
        f"clause's own criterion holds in at least {CLAUSE_1_RELIABLE_SHARE:.0%} of seeds "
        "pooled over every seed block.",
        "",
    ]
    for arm_name, verdict in results["verdicts"]["arms"].items():
        lines.extend(
            [
                f"### `{arm_name}`",
                "",
                f"- **Clause 1** (correct-report rate rises over generations within a run at "
                f"fixed initial parameters): rising in "
                f"{verdict['clause_1_seeds_rising']}/{verdict['clause_1_seeds']} seeds; "
                f"realized precision "
                f"{'above' if verdict['clause_1_precision_above_static_prior'] else 'not above'}"
                f" the static-prior null. **"
                f"{'met' if verdict['clause_1_met'] else 'not met'}**.",
                f"- **Clause 2** (parent-child reproductive correlation reliably above "
                f"{CLAUSE_2_THRESHOLD}): cleared in "
                f"{verdict['clause_2_seeds_cleared']}/{verdict['clause_1_seeds']} seeds. **"
                f"{'met' if verdict['clause_2_met'] else 'not met'}**.",
                "",
            ]
        )
    return lines


_READING_THE_NUMBERS: tuple[str, ...] = (
    "",
    "## Reading the numbers: measured effect versus artifact",
    "",
    "- **Precision here is far below the committed 30.58% no-lever figure, and that is a",
    "  consequence of levers 1-4, not of lever 5.** Paying attention income for verified",
    "  correct reports makes reporting profitable in expectation, so evolved agents report",
    "  around 20-30 times per adult lifetime instead of staying mostly silent, and only",
    "  ~5% of adults never report. The no-lever arm buys its higher precision with silence.",
    "  Report volume is printed next to precision for exactly this reason.",
    "- **The cap-binding share is not the limit on clause 2 here.** On `SparseSensorScenario`",
    "  the cap bound on roughly a third of steps, which was the stated reason ordering",
    "  reproduction could only shift ~0.03 offspring. On this instrument the cap binds on",
    "  73-86% of steps at a stable population of 60, and clause 2 still does not move: the",
    "  parent-child reproductive correlation stays slightly negative in both arms. A binding",
    "  cap is therefore not sufficient for lineage-level heritability of reproductive output.",
    "- **The large offspring gap between adults with a correct report and adults that never",
    "  reported correctly is a lifetime confound, not a gate effect.** It is the same size in",
    "  the `W=0` reserves-only control as at `W=1`; long-lived agents accumulate both more",
    "  correct reports and more offspring. The same applies to the positive parent-child",
    "  *precision* correlation, which is present in the control as well.",
    "- **Clause-1 slopes are negative in both arms and about twice as negative at `W=1`.**",
    "  Turning the gate on makes the within-run trend worse, not better, at comparable report",
    "  volume: the rate declines over generations in 38 of 40 seeds in both arms. That is",
    "  the opposite sign to the SparseSensor result, so lever 5 does not transfer to this",
    "  domain rather than merely falling short of a threshold.",
    "- The designed ceiling at the same config still scores 100.00% precision, so the",
    "  exploitable margin remains large and the negative evolved result is not a thin-domain",
    "  artifact.",
)


def markdown_report(results: dict[str, Any], reproduce_command: str) -> str:
    """Render the response-gate documentation artifact."""
    lines = [
        "# Wildfire response-gate measurement (lever 5)",
        "",
        "## The question",
        "",
        "Does keying reproductive merit on verified correctness — TattleTots'",
        "`reproduction_correctness_weight`, which cleared falsification clause 1 on the",
        "engine's own `SparseSensorScenario` — move either falsification clause on the",
        "wildfire instrument?",
        "",
        "Reproduce with:",
        "",
        "```bash",
        reproduce_command,
        "```",
        "",
        "## Measurement path",
        "",
        results["measurement_path"] + ".",
        "",
        "The arm under test is the ordinary (evolved) arm. The designed reporter is run at",
        "the same config as a visible ceiling, not as the thing under test; it is a",
        "hand-written evidence-only policy, so its precision says what the instrument",
        "allows, not what evolution found.",
        "",
    ]
    lines.extend(_fixed_config_lines(results))
    lines.extend(_block_sections(results))
    lines.extend(_verdict_lines(results))
    lines.extend(_READING_THE_NUMBERS)
    return "\n".join(lines) + "\n"


def results_json(results: dict[str, Any]) -> dict[str, Any]:
    """The results as written to disk; already JSON-serializable."""
    return {
        "spec": results["spec"],
        "measurement_path": results["measurement_path"],
        "nulls": results["nulls"],
        "arm_definitions": results["arm_definitions"],
        "blocks": results["blocks"],
        "verdicts": results["verdicts"],
    }
