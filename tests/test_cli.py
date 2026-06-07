"""Tests for CLI entrypoint."""

from __future__ import annotations

from fire_ecology.cli import main


class TestCLI:
    def test_basic_run(self) -> None:
        main(["sim", "--steps", "10", "--grid-rows", "5", "--grid-cols", "5"])

    def test_json_output(self, capsys: object) -> None:
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
