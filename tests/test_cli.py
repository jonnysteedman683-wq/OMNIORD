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


def test_run_reports_when_no_model_available() -> None:
    # With no local Ollama server and no cloud key, the router has no tier;
    # the command should fail gracefully rather than crash.
    result = runner.invoke(app, ["run", "do a thing"])
    assert result.exit_code == 1
    assert "No model available" in result.output


def test_run_rejects_bad_tier() -> None:
    result = runner.invoke(app, ["run", "do a thing", "--tier", "bogus"])
    assert result.exit_code == 2


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("version", "config", "run"):
        assert command in result.output
