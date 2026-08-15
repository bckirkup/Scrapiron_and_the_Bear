"""Re-measure Fire comparison arms and instrument validation."""

from __future__ import annotations

import contextlib
import io
import json
import sys

from tattletots.interface.instrument import validate_instrument

from fire_ecology.adapter.fire_adapter import FireEcologyAdapter
from fire_ecology.cli import main as cli_main


def _cli_json(argv: list[str]) -> object:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli_main(argv)
    return json.loads(buf.getvalue())


def _scalars(row: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in row.items() if not isinstance(v, (list, dict))}


def main() -> int:
    out: dict[str, object] = {}

    rows = _cli_json(["compare", "--steps", "100", "--seed", "42", "--json"])
    assert isinstance(rows, list)
    out["comparison"] = {str(r["architecture"]): _scalars(r) for r in rows}

    abl_rows = _cli_json(
        ["compare", "--steps", "100", "--seed", "42", "--a4-opir-ablation", "--json"]
    )
    assert isinstance(abl_rows, list)
    out["comparison_with_ablation"] = {str(r["architecture"]): _scalars(r) for r in abl_rows}

    report = validate_instrument(FireEcologyAdapter(), steps=200)
    out["instrument_validation"] = {
        "findings": [str(f) for f in report.findings],
        "measured_steps": report.measured_steps,
        "event_steps": report.event_steps,
        "distinct_event_locations": report.distinct_event_locations,
        "inferability_precision": report.inferability_precision,
        "decoder_precision": report.decoder_precision,
        "static_prior_baseline": report.static_prior_baseline,
        "chance_baseline": report.chance_baseline,
        "candidate_locations": report.candidate_locations,
    }

    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
