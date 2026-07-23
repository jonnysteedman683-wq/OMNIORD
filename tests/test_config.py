"""Phase 1 tests: configuration loading, defaults, and env overrides."""

from __future__ import annotations

import pytest

from omniord.config import CloudTierConfig, OmniordSettings, get_settings


def test_defaults_are_local_first() -> None:
    settings = OmniordSettings(_env_file=None)
    assert settings.prefer_local is True
    assert settings.max_retries == 3
    assert settings.local.base_url == "http://localhost:11434"
    assert settings.local.fast_model == "llama3.1"
    assert settings.local.code_model == "qwen2.5-coder"
    assert settings.cloud.provider == "anthropic"


def test_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIORD_LOCAL__FAST_MODEL", "llama3.2")
    monkeypatch.setenv("OMNIORD_CLOUD__PROVIDER", "openai")
    monkeypatch.setenv("OMNIORD_MAX_RETRIES", "5")

    settings = OmniordSettings(_env_file=None)

    assert settings.local.fast_model == "llama3.2"
    assert settings.cloud.provider == "openai"
    assert settings.max_retries == 5


def test_cloud_availability_tracks_selected_provider() -> None:
    without_key = CloudTierConfig(provider="anthropic")
    assert without_key.is_available is False

    anthropic = CloudTierConfig(provider="anthropic", anthropic_api_key="sk-ant-x")
    assert anthropic.is_available is True
    assert anthropic.active_model == anthropic.anthropic_model
    assert anthropic.active_key == "sk-ant-x"

    # A key for the *other* provider does not make the selected one available.
    mismatched = CloudTierConfig(provider="openai", anthropic_api_key="sk-ant-x")
    assert mismatched.is_available is False


def test_validation_rejects_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIORD_LOCAL__CONFIDENCE_THRESHOLD", "2.0")
    with pytest.raises(ValueError):
        OmniordSettings(_env_file=None)


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()
