"""Provider selection (Phase 8D): `get_provider` is the one factory, `local` is the default, nothing falls back silently."""

from pathlib import Path

import pytest

from src.data_loader import (
    KNOWN_PROVIDERS,
    DataProvider,
    LocalDataProvider,
    ProviderConfigurationError,
    ProviderNotImplementedError,
    get_provider,
    normalize_provider_name,
)

CONFIG = {"data": {"provider": "local", "prices_file": "data/raw/prices.csv"}}
ROOT = Path("project_root")  # never touched on disk: the factory only builds a path


def test_the_known_providers_are_local_and_angel_one():
    assert KNOWN_PROVIDERS == ("local", "angel_one")


@pytest.mark.parametrize("raw, expected", [("local", "local"), ("  LOCAL ", "local"), ("angel_one", "angel_one"),
                                           ("Angel_One", "angel_one"), ("angelone", "angel_one"), ("", ""), (None, "")])
def test_provider_names_are_normalised(raw, expected):
    assert normalize_provider_name(raw) == expected


# ------------------------------------------------------------------ local is the default
def test_local_is_the_default_when_nothing_is_configured():
    provider = get_provider({"data": {"prices_file": "a.csv"}}, ROOT)
    assert isinstance(provider, LocalDataProvider) and isinstance(provider, DataProvider)


def test_config_yaml_chooses_the_provider_when_there_is_no_override():
    assert isinstance(get_provider(CONFIG, ROOT), LocalDataProvider)


@pytest.mark.parametrize("override", [None, "", "   ", "local", "LOCAL", " Local "])
def test_local_override_values_build_the_local_provider(override):
    assert isinstance(get_provider(CONFIG, ROOT, provider=override), LocalDataProvider)


def test_the_override_beats_config_yaml():
    with pytest.raises(NotImplementedError):
        get_provider(CONFIG, ROOT, provider="angel_one")  # config says local, the override wins and is refused


# ------------------------------------------------------------------ where the local file is
def test_a_relative_price_file_starts_at_the_base_directory():
    assert get_provider(CONFIG, ROOT).path == ROOT / "data/raw/prices.csv"


def test_data_path_overrides_the_configured_price_file_and_starts_at_the_base_directory():
    assert get_provider(CONFIG, ROOT, data_path="elsewhere/other.csv").path == ROOT / "elsewhere/other.csv"


def test_an_absolute_data_path_is_used_as_given(tmp_path):
    assert get_provider(CONFIG, ROOT, data_path=tmp_path / "x.csv").path == tmp_path / "x.csv"


def test_an_empty_data_path_means_use_config_yaml():
    assert get_provider(CONFIG, ROOT, data_path="").path == ROOT / "data/raw/prices.csv"


# ------------------------------------------------------------------ angel_one: represented, not implemented
@pytest.mark.parametrize("name", ["angel_one", "ANGEL_ONE", "angelone", " Angel_One "])
def test_angel_one_is_refused_with_a_clear_message_and_never_replaced_by_local_data(name):
    with pytest.raises(ProviderNotImplementedError) as caught:
        get_provider(CONFIG, ROOT, provider=name)
    message = str(caught.value)
    assert "not implemented" in message and "Phase 8E" in message and "DATA_PROVIDER=local" in message
    assert isinstance(caught.value, NotImplementedError)


def test_angel_one_named_in_config_yaml_is_refused_too():
    with pytest.raises(ProviderNotImplementedError):
        get_provider({"data": {"provider": "angel_one", "prices_file": "a.csv"}}, ROOT)


# ------------------------------------------------------------------ unknown providers
@pytest.mark.parametrize("name", ["unknown", "yahoo", "alpaca", "local2", "angel one"])
def test_an_unknown_provider_fails_clearly_and_lists_the_supported_ones(name):
    with pytest.raises(ProviderConfigurationError) as caught:
        get_provider(CONFIG, ROOT, provider=name)
    message = str(caught.value)
    assert "Unknown data provider" in message and "local" in message and "angel_one" in message
    assert isinstance(caught.value, ValueError)


def test_a_very_long_unknown_name_is_shortened_in_the_message():
    with pytest.raises(ProviderConfigurationError) as caught:
        get_provider(CONFIG, ROOT, provider="x" * 500)
    assert len(str(caught.value)) < 200


def test_no_credentials_are_needed_or_read_to_build_the_local_provider(monkeypatch):
    for name in ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PIN", "ANGEL_TOTP_SECRET"):
        monkeypatch.delenv(name, raising=False)
    assert isinstance(get_provider(CONFIG, ROOT), LocalDataProvider)
