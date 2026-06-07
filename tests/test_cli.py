"""Tests for CLI entrypoint."""

from __future__ import annotations

from fire_ecology.cli import main


class TestCLI:
    def test_basic_run(self) -> None:
        main(["--steps", "10", "--grid-rows", "5", "--grid-cols", "5"])

    def test_json_output(self, capsys: object) -> None:
        main(["--steps", "5", "--grid-rows", "5", "--grid-cols", "5", "--json"])

    def test_verbose(self) -> None:
        main(["--steps", "10", "--grid-rows", "5", "--grid-cols", "5", "--verbose"])
