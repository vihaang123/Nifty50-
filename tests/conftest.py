"""Shared test setup.

Every test runs against the local synthetic dataset, whatever is set in the shell that launched pytest. This keeps the
suite deterministic and guarantees that no test can reach a real market-data provider (Angel One does not exist in this
codebase yet, and this fixture would keep it out of the tests even when it does).

A test that needs a different value sets it itself with monkeypatch.setenv; that runs after this fixture and wins.
"""

import pytest


@pytest.fixture(autouse=True)
def local_data_provider_only(monkeypatch):
    monkeypatch.setenv("DATA_PROVIDER", "local")
    monkeypatch.delenv("DATA_PATH", raising=False)
    for name in ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PIN", "ANGEL_TOTP_SECRET"):
        monkeypatch.delenv(name, raising=False)
