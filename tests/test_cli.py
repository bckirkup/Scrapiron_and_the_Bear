"""Tests for CLI entrypoint."""

from __future__ import annotations

import json

import pytest

from fire_ecology.cli import main


class TestCLI:
    def test_basic_run(self) -> None:
        main(["sim", "--steps", "10", "--grid-rows", "5", "--grid-cols", "5"])

    def test_json_output(self) -> None:
        main(["sim", "--steps", "5", "--grid-rows", "5", "--grid-cols", "5", "--json"])

    def test_verbose(self) -> None:
        main(["sim", "--steps", "10", "--grid-rows", "5", "--grid-cols", "5", "--verbose"])

    def test_default_subcommand(self) -> None:
        """No subcommand falls through to sim."""
        main(["--steps", "5", "--grid-rows", "5", "--grid-cols", "5"])

    def test_compare_no_a4(self) -> None:
        main(
            [
                "compare",
                "--steps",
                "10",
                "--grid-rows",
                "5",
                "--grid-cols",
                "5",
                "--n-drones",
                "3",
                "--no-a4",
            ]
        )

    def test_compare_json(self) -> None:
        main(
            [
                "compare",
                "--steps",
                "10",
                "--grid-rows",
                "5",
                "--grid-cols",
                "5",
                "--n-drones",
                "3",
                "--no-a4",
                "--json",
            ]
        )

    def test_compare_ablates_the_opir_backstop(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``--a4-ablate-opir-backstop`` makes the A4 arm agent-only."""
        main(
            [
                "compare",
                "--steps",
                "12",
                "--grid-rows",
                "10",
                "--grid-cols",
                "10",
                "--n-drones",
                "3",
                "--a4-ablate-opir-backstop",
                "--json",
            ]
        )
        a4 = next(
            arm for arm in json.loads(capsys.readouterr().out) if arm["architecture"] == "A4 BMA"
        )

        assert a4["opir_backstop_ablated"] is True
        assert a4["opir_detections"] == 0
        assert a4["opir_shadow_detections"] > 0
        assert a4["tot_detection_share"] == pytest.approx(1.0)

    def test_compare_keeps_the_backstop_by_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "compare",
                "--steps",
                "12",
                "--grid-rows",
                "10",
                "--grid-cols",
                "10",
                "--n-drones",
                "3",
                "--json",
            ]
        )
        a4 = next(
            arm for arm in json.loads(capsys.readouterr().out) if arm["architecture"] == "A4 BMA"
        )

        assert a4["opir_backstop_ablated"] is False
        assert a4["opir_detections"] > 0
        assert a4["opir_shadow_detections"] == 0

    def test_main_reads_sys_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The installed entrypoint takes its flags from ``sys.argv``."""
        monkeypatch.setattr(
            "sys.argv",
            ["fire-ecology", "sim", "--steps", "3", "--grid-rows", "5", "--grid-cols", "5"],
        )
        main()
