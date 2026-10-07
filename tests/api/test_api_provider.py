"""Phase 8D: provider selection, provider status, cache identity and production-safe errors through the API."""

import shutil
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from api import services
from api.main import create_app
from api.source import DataSource

from conftest import ROOT, strict_json

DATA_ENDPOINTS = [
    ("get", "/api/dataset"),
    ("get", "/api/universe"),
    ("get", "/api/analysis/pca"),
    ("get", "/api/analysis/lda"),
    ("get", "/api/similarity/AXISBANK"),
    ("post", "/api/basket/generate"),
    ("post", "/api/backtest"),
]
LEAKS = ("Traceback", "/home/", "/tmp/", "Users/", "\\Users\\", ".csv", ".py", "FileNotFoundError", "RuntimeError")


def call(client, method, url):
    return client.get(url) if method == "get" else client.post(url, json={})


def fresh_client():
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def isolated_caches():
    services.clear_caches()
    yield
    services.clear_caches()


# ===================================================== the default: local, synthetic
def test_dataset_reports_the_local_synthetic_provider():
    body = fresh_client().get("/api/dataset").json()
    assert body["provider"] == "local" and body["source"] == "synthetic" and body["is_synthetic"] is True
    assert body["observations"] == 20880 == body["stock_count"] * body["trading_days"]


def test_explicit_local_behaves_exactly_like_the_default(monkeypatch):
    default = fresh_client().get("/api/dataset").json()
    monkeypatch.setenv("DATA_PROVIDER", "LOCAL")
    assert fresh_client().get("/api/dataset").json() == default


def test_health_reports_status_environment_and_data_provider():
    body = fresh_client().get("/api/health").json()
    assert body["status"] == "ok" and body["environment"] == "development"
    assert body["data"] == {"provider": "local", "available": True, "is_synthetic": True, "message": None}


def test_health_reports_production_when_the_environment_says_so(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.example.com")
    assert fresh_client().get("/api/health").json()["environment"] == "production"


def test_health_does_not_load_the_dataset():
    fresh_client().get("/api/health")
    assert services.get_dataset.cache_info().currsize == 0  # a health check stays cheap


# ===================================================== angel_one: represented, not implemented
@pytest.mark.parametrize("name", ["angel_one", "ANGEL_ONE", "angelone"])
@pytest.mark.parametrize("method, url", DATA_ENDPOINTS)
def test_angel_one_gives_a_clear_503_on_every_data_endpoint_and_never_local_data(monkeypatch, name, method, url):
    monkeypatch.setenv("DATA_PROVIDER", name)
    response = call(fresh_client(), method, url)
    assert response.status_code == 503
    error = strict_json(response)["error"]
    assert error["status"] == 503 and error["code"] == "data_provider_not_implemented"
    assert "not implemented" in error["message"] and "Phase 8E" in error["message"]
    assert "stocks" not in response.json() and "summary" not in response.json()  # no data was served
    for leak in LEAKS:
        assert leak not in response.text, leak


def test_health_says_the_angel_one_provider_is_unavailable_instead_of_failing(monkeypatch):
    monkeypatch.setenv("DATA_PROVIDER", "angel_one")
    response = fresh_client().get("/api/health")
    body = response.json()
    assert response.status_code == 200 and body["status"] == "degraded"
    assert body["data"]["provider"] == "angel_one" and body["data"]["available"] is False
    assert body["data"]["is_synthetic"] is None and "not implemented" in body["data"]["message"]


# ===================================================== unknown providers
@pytest.mark.parametrize("method, url", DATA_ENDPOINTS)
def test_an_unknown_provider_gives_a_clear_503(monkeypatch, method, url):
    monkeypatch.setenv("DATA_PROVIDER", "unknown")
    response = call(fresh_client(), method, url)
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "data_provider_not_configured"
    assert "Unknown data provider 'unknown'" in error["message"] and "local" in error["message"] and "angel_one" in error["message"]


def test_the_api_still_starts_with_a_wrong_provider_and_health_explains_it(monkeypatch):
    monkeypatch.setenv("DATA_PROVIDER", "unknown")
    body = fresh_client().get("/api/health").json()  # create_app() must not raise
    assert body["status"] == "degraded" and body["data"]["provider"] == "unknown" and body["data"]["available"] is False
    assert "Unknown data provider" in body["data"]["message"]


def test_a_very_long_wrong_provider_name_is_shortened_everywhere(monkeypatch):
    monkeypatch.setenv("DATA_PROVIDER", "y" * 400)
    client = fresh_client()
    assert len(client.get("/api/dataset").text) < 400
    assert len(client.get("/api/health").text) < 600


# ===================================================== unavailable or invalid data
def test_a_missing_data_file_is_a_503_that_does_not_reveal_the_path(monkeypatch, tmp_path):
    secret_folder = tmp_path / "very_secret_folder"
    monkeypatch.setenv("DATA_PATH", str(secret_folder / "missing_prices.csv"))
    for method, url in DATA_ENDPOINTS:
        response = call(fresh_client(), method, url)
        assert response.status_code == 503 and response.json()["error"]["code"] == "data_unavailable", url
        for leak in ("very_secret_folder", "missing_prices", str(tmp_path), *LEAKS):
            assert leak not in response.text, (url, leak)


def test_health_reports_missing_data_without_the_path(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "very_secret_folder" / "missing.csv"))
    response = fresh_client().get("/api/health")
    body = response.json()
    assert response.status_code == 200 and body["status"] == "degraded"
    assert body["data"]["provider"] == "local" and body["data"]["available"] is False
    assert "very_secret_folder" not in response.text and str(tmp_path) not in response.text


def _write_prices(path: Path, mutate):
    import pandas as pd

    frame = pd.read_csv(ROOT / "data/raw/dev_prices_synthetic.csv")
    mutate(frame)
    frame.to_csv(path, index=False)


def test_impossible_data_is_refused_with_a_503_and_nothing_is_served(monkeypatch, tmp_path):
    bad = tmp_path / "bad_prices.csv"
    _write_prices(bad, lambda f: f.__setitem__("high", f["low"] - 1))  # high < low on every row
    monkeypatch.setenv("DATA_PATH", str(bad))
    response = fresh_client().get("/api/dataset")
    assert response.status_code == 503 and response.json()["error"]["code"] == "data_invalid"
    assert "failed validation" in response.json()["error"]["message"]
    for leak in (str(tmp_path), "bad_prices", *LEAKS):
        assert leak not in response.text


# ===================================================== DATA_PATH
def test_data_path_switches_the_data_the_api_serves(monkeypatch, tmp_path):
    smaller = tmp_path / "three_stocks_synthetic.csv"
    _write_prices(smaller, lambda f: f.drop(f[~f["symbol"].isin(["TCS", "INFY", "ITC"])].index, inplace=True))
    monkeypatch.setenv("DATA_PATH", str(smaller))
    body = fresh_client().get("/api/dataset").json()
    assert body["stocks"] == ["INFY", "ITC", "TCS"] and body["provider"] == "local" and body["is_synthetic"] is True


def test_a_data_path_without_synthetic_in_its_name_is_not_labelled_synthetic(monkeypatch, tmp_path):
    other = tmp_path / "market_prices.csv"
    shutil.copy(ROOT / "data/raw/dev_prices_synthetic.csv", other)
    monkeypatch.setenv("DATA_PATH", str(other))
    body = fresh_client().get("/api/dataset").json()
    assert body["is_synthetic"] is False and body["source"] == "local" and "synthetic" not in body["notice"].lower()


def test_a_relative_data_path_starts_at_the_directory_of_the_config_file(monkeypatch, tmp_path):
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    (tmp_path / "files").mkdir()
    shutil.copy(ROOT / config["data"]["prices_file"], tmp_path / "files" / "relative_synthetic.csv")
    shutil.copy(ROOT / config["data"]["market_index_file"], tmp_path / "market.csv")
    config["data"].update(market_index_file="market.csv")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("DATA_PATH", "files/relative_synthetic.csv")
    assert fresh_client().get("/api/dataset").status_code == 200


# ===================================================== caching never crosses data configurations
def test_every_cache_is_keyed_by_the_full_data_source():
    config = str(ROOT / "config.yaml")
    local = DataSource(config)
    assert DataSource(config) == local and hash(DataSource(config)) == hash(local)
    assert DataSource(config, provider="local") != local
    assert DataSource(config, data_path="x.csv") != local
    assert DataSource(config + ".other") != local


def test_different_data_paths_do_not_share_cached_objects(tmp_path):
    config = str(ROOT / "config.yaml")
    smaller = tmp_path / "two_synthetic.csv"
    _write_prices(smaller, lambda f: f.drop(f[~f["symbol"].isin(["TCS", "INFY"])].index, inplace=True))
    default, other = DataSource(config), DataSource(config, data_path=str(smaller))
    a, b = services.get_dataset(default), services.get_dataset(other)
    assert a is not b and a.stocks["symbol"].nunique() == 10 and b.stocks["symbol"].nunique() == 2
    assert services.get_dataset(default) is a and services.get_dataset(other) is b  # and each is still cached
    assert services.get_lda(default) is not services.get_lda(other)
    assert services.get_similarity(default) is not services.get_similarity(other)
    assert services.get_pca(default, 5) is not services.get_pca(other, 5)


def test_switching_the_data_path_and_back_never_serves_stale_results(monkeypatch, tmp_path):
    smaller = tmp_path / "two_synthetic.csv"
    _write_prices(smaller, lambda f: f.drop(f[~f["symbol"].isin(["TCS", "INFY"])].index, inplace=True))
    client = fresh_client()
    assert client.get("/api/dataset").json()["stock_count"] == 10
    monkeypatch.setenv("DATA_PATH", str(smaller))
    assert client.get("/api/dataset").json()["stock_count"] == 2
    monkeypatch.delenv("DATA_PATH")
    assert client.get("/api/dataset").json()["stock_count"] == 10


def test_a_failed_provider_is_not_cached_so_fixing_it_works_without_a_restart(monkeypatch):
    client = fresh_client()
    monkeypatch.setenv("DATA_PROVIDER", "angel_one")
    assert client.get("/api/dataset").status_code == 503
    monkeypatch.setenv("DATA_PROVIDER", "local")
    assert client.get("/api/dataset").status_code == 200


# ===================================================== error handling: the status-code table
def test_status_codes_for_each_kind_of_failure(monkeypatch):
    client = fresh_client()
    assert client.get("/api/similarity/AXISBANK?top_n=0").status_code == 422     # invalid input
    assert client.post("/api/backtest", json={"capital": -1}).status_code == 422  # invalid input
    assert client.get("/api/similarity/NOPE").status_code == 404                  # unknown resource
    assert client.get("/api/not-a-route").status_code == 404
    assert client.post("/api/basket/generate", json={"basket_size": 99}).status_code == 400  # valid request, impossible basket
    monkeypatch.setenv("DATA_PROVIDER", "angel_one")
    assert client.get("/api/dataset").status_code == 503                          # configuration unavailable


def test_every_error_has_the_one_documented_shape(monkeypatch):
    client = fresh_client()
    monkeypatch.setenv("DATA_PROVIDER", "nope")
    for response in (client.get("/api/dataset"), client.get("/api/similarity/AXISBANK"), client.get("/api/missing")):
        error = response.json()["error"]
        assert set(error) >= {"status", "code", "message"} and error["status"] == response.status_code


def test_openapi_documents_the_503(client):
    paths = client.get("/openapi.json").json()["paths"]
    for url in ("/api/dataset", "/api/universe", "/api/analysis/pca", "/api/analysis/lda", "/api/similarity/{symbol}",
                "/api/basket/generate", "/api/backtest"):
        operation = next(iter(paths[url].values()))
        assert "503" in operation["responses"], url


def test_unexpected_errors_in_production_are_still_clean(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.example.com")

    def explode(source):
        raise RuntimeError("boom at /srv/app/api/services.py password=hunter2")

    monkeypatch.setattr(services, "dataset_info", explode)
    response = fresh_client().get("/api/dataset")
    assert response.status_code == 500
    assert response.json() == {"error": {"status": 500, "code": "internal_error", "message": "Unexpected internal error. Please try again later."}}
    assert "hunter2" not in response.text and "/srv/" not in response.text


# ===================================================== no Angel One anywhere
def test_no_source_file_imports_or_calls_an_angel_one_client():
    forbidden = ("smartapi", "smartconnect", "angelbroking", "angelone.in", "angelbroking.com", "pyotp", "import requests", "import httpx", "urllib.request")
    for folder in ("src", "api"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for word in forbidden:
                assert word not in text, f"{path.relative_to(ROOT)} mentions {word}"
    assert "smartapi" not in (ROOT / "requirements.txt").read_text().lower().replace("# smartapi-python  # phase 8e: angel one sdk", "")


def test_the_angel_one_credentials_are_never_read_by_the_code():
    names = ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PIN", "ANGEL_TOTP_SECRET")
    for folder in ("src", "api"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for name in names:
                assert name not in text, f"{path.relative_to(ROOT)} reads {name}"


# ===================================================== no machine-specific paths in the runtime code
def test_the_runtime_code_has_no_machine_specific_paths():
    markers = ("/home/", "/tmp/", "/Users/", "/root/", "C:\\", "C:/", "/mnt/", "/var/")
    runtime = [ROOT / "index.py", ROOT / "config.yaml", *(ROOT / "api").rglob("*.py"), *(ROOT / "src").rglob("*.py")]
    for path in runtime:
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            assert marker not in text, f"{path.relative_to(ROOT)} contains {marker!r}"
