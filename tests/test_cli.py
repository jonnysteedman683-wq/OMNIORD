"""Phase 1 tests: the Typer CLI surface."""

from __future__ import annotations

from typer.testing import CliRunner

from omniord import __version__
from omniord.main import app

runner = CliRunner()


def test_no_args_shows_banner() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Omniord" in result.output or "Omniord".lower() in result.output.lower()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_config_command_lists_tiers() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "local.fast_model" in result.output
    assert "cloud.provider" in result.output


def test_run_is_recognized_but_not_yet_implemented() -> None:
    result = runner.invoke(app, ["run", "do a thing"])
    assert result.exit_code == 0
    assert "do a thing" in result.output
    assert "not implemented" in result.output.lower()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("version", "config", "run"):
        assert command in result.output
