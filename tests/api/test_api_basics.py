"""API basics: health, docs, error shape, CORS, settings, serialisation, architecture."""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.routes.data as data_routes
from api import API_VERSION, SERVICE_NAME
from api.main import create_app
from api.serialization import to_jsonable
from api.settings import get_settings, parse_origins

from conftest import ROOT, strict_json  # noqa: E402  (tests/api is on sys.path)

ORIGIN = "http://localhost:3000"


# ===================== health =====================
def test_health_returns_200_and_the_exact_body(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "stock-basket-api", "version": "1.0.0"}
    assert SERVICE_NAME == "stock-basket-api" and API_VERSION == "1.0.0"
    assert response.headers["content-type"].startswith("application/json")


def test_health_only_allows_get(client):
    response = client.post("/api/health")
    assert response.status_code == 405 and response.json()["error"]["code"] == "method_not_allowed"


# ===================== docs =====================
def test_swagger_docs_and_redoc_work(client):
    docs, redoc = client.get("/docs"), client.get("/redoc")
    assert docs.status_code == 200 and "swagger" in docs.text.lower()
    assert redoc.status_code == 200 and "redoc" in redoc.text.lower()


def test_openapi_lists_every_endpoint_with_a_response_schema(client):
    spec = client.get("/openapi.json").json()
    expected = {
        ("get", "/api/health"), ("get", "/api/dataset"), ("get", "/api/universe"), ("get", "/api/analysis/pca"),
        ("get", "/api/analysis/lda"), ("get", "/api/similarity/{symbol}"), ("post", "/api/basket/generate"), ("post", "/api/backtest"),
    }
    found = {(method, path) for path, item in spec["paths"].items() for method in item}
    assert found == expected
    for path, method in [(p, m) for m, p in expected]:
        operation = spec["paths"][path][method]
        assert operation["summary"] and operation["description"] if path != "/api/health" else operation["summary"]
        schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
        assert "$ref" in schema
    assert spec["info"]["title"] == "Stock Basket API" and spec["info"]["version"] == "1.0.0"
    assert "synthetic" in spec["info"]["description"].lower() and "not investment advice" in spec["info"]["description"].lower()


def test_error_responses_are_documented_in_the_contract(client):
    spec = client.get("/openapi.json").json()
    assert "404" in spec["paths"]["/api/similarity/{symbol}"]["get"]["responses"]
    assert "400" in spec["paths"]["/api/basket/generate"]["post"]["responses"]
    for path, item in spec["paths"].items():
        for operation in item.values():
            if "422" in operation["responses"]:
                assert operation["responses"]["422"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")


# ===================== error handling =====================
def test_unknown_route_is_a_clean_404(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.json() == {"error": {"status": 404, "code": "not_found", "message": "The requested resource was not found."}}


def test_validation_errors_use_the_common_shape_and_do_not_echo_input(client):
    response = client.post("/api/basket/generate", json={"capital": -5, "basket_size": 10, "secret": "hunter2"})
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["status"] == 422 and body["code"] == "invalid_request" and body["message"] == "Invalid request parameters."
    fields = {d["field"] for d in body["details"]}
    assert fields == {"capital", "secret"} and all(set(d) == {"field", "message"} for d in body["details"])
    assert "hunter2" not in response.text and "-5" not in response.text


def test_an_unexpected_error_is_a_generic_500_without_a_traceback(client, monkeypatch):
    def explode(path):
        raise RuntimeError("secret /home/claude/project/api/services.py exploded with password=hunter2")

    monkeypatch.setattr(data_routes.services, "dataset_info", explode)
    response = client.get("/api/dataset")
    assert response.status_code == 500
    assert response.json() == {"error": {"status": 500, "code": "internal_error", "message": "Unexpected internal error. Please try again later."}}
    for leak in ("Traceback", "hunter2", "services.py", "RuntimeError", "/home/"):
        assert leak not in response.text


# ===================== CORS =====================
def test_preflight_from_the_dev_frontend_is_allowed(client):
    response = client.options(
        "/api/basket/generate",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "access-control-allow-credentials" not in response.headers


def test_a_normal_request_from_the_frontend_gets_the_cors_header(client):
    response = client.get("/api/health", headers={"Origin": ORIGIN})
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_other_origins_are_not_allowed(client):
    response = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers
    preflight = client.options("/api/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert preflight.status_code == 400 and "access-control-allow-origin" not in preflight.headers


def test_the_allowed_origin_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.example.com, https://staging.example.com/")
    app = TestClient(create_app(), raise_server_exceptions=False)
    for origin in ("https://app.example.com", "https://staging.example.com"):
        assert app.get("/api/health", headers={"Origin": origin}).headers["access-control-allow-origin"] == origin
    assert "access-control-allow-origin" not in app.get("/api/health", headers={"Origin": ORIGIN}).headers


# ===================== settings =====================
def test_default_settings_are_the_development_ones(monkeypatch):
    for name in ("FRONTEND_ORIGIN", "ENVIRONMENT", "CONFIG_PATH"):
        monkeypatch.delenv(name, raising=False)
    s = get_settings()
    assert s.environment == "development" and s.frontend_origins == (ORIGIN,) and s.config_path == ROOT / "config.yaml"


def test_production_refuses_a_wildcard_origin(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FRONTEND_ORIGIN", "*")
    with pytest.raises(RuntimeError, match="'\\*' is not allowed"):
        get_settings()
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.example.com,*")
    with pytest.raises(RuntimeError, match="not allowed"):
        get_settings()


def test_production_needs_an_explicit_origin(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("FRONTEND_ORIGIN", raising=False)
    with pytest.raises(RuntimeError, match="must be set"):
        get_settings()
    monkeypatch.setenv("FRONTEND_ORIGIN", "https://app.example.com")
    assert get_settings().frontend_origins == ("https://app.example.com",)


def test_a_bad_environment_or_origin_is_refused(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    with pytest.raises(RuntimeError, match="ENVIRONMENT"):
        get_settings()
    monkeypatch.setenv("ENVIRONMENT", "development")
    for bad in ("localhost:3000", "http://", "javascript:alert(1)", "http://a.com/path"):
        monkeypatch.setenv("FRONTEND_ORIGIN", bad)
        with pytest.raises(RuntimeError, match="not an origin"):
            get_settings()


def test_parse_origins_cleans_and_deduplicates():
    assert parse_origins(" http://a.com/ ,http://a.com, http://b.com:8080 ,") == ("http://a.com", "http://b.com:8080")


# ===================== JSON safety =====================
def test_to_jsonable_converts_pandas_and_numpy_and_removes_nan():
    frame = pd.DataFrame({"d": pd.to_datetime(["2024-01-02", "2024-01-03"]), "x": [np.float64(1.5), np.nan], "n": np.array([1, 2], dtype=np.int64)})
    out = to_jsonable({"frame": frame, "inf": np.float64("inf"), "i": np.int64(3), "t": pd.Timestamp("2024-05-06"), "b": np.bool_(True), "nan": float("nan")})
    assert out == {
        "frame": [{"d": "2024-01-02", "x": 1.5, "n": 1}, {"d": "2024-01-03", "x": None, "n": 2}],
        "inf": None, "i": 3, "t": "2024-05-06", "b": True, "nan": None,
    }
    json.dumps(out, allow_nan=False)  # strict JSON accepts it
    assert type(out["i"]) is int and type(out["frame"][0]["n"]) is int


@pytest.mark.parametrize("url", ["/api/health", "/api/dataset", "/api/universe", "/api/analysis/pca", "/api/analysis/lda", "/api/similarity/AXISBANK"])
def test_every_get_response_is_strict_json(client, url):
    strict_json(client.get(url))


# ===================== architecture =====================
API_FILES = sorted((ROOT / "api").rglob("*.py"))


def _imports(path: Path) -> set:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
    return found


def test_the_api_has_no_angel_one_code_or_credentials():
    assert API_FILES
    for path in API_FILES:
        assert not [m for m in _imports(path) if "angel" in m.lower() or "smartapi" in m.lower() or "pyotp" in m.lower()], path
        text = path.read_text(encoding="utf-8")
        for secret in ("ANGEL_", "API_KEY", "TOTP", "PIN="):
            assert secret not in text, (path, secret)


def test_routes_are_thin_and_contain_no_research_logic():
    for path in sorted((ROOT / "api" / "routes").glob("*.py")):
        banned = [m for m in _imports(path) if m.split(".")[0] in {"pandas", "numpy", "sklearn", "src", "matplotlib"}]
        assert not banned, (path, banned)


def test_only_the_service_layer_imports_the_research_engine():
    for path in API_FILES:
        uses_src = any(m.split(".")[0] == "src" for m in _imports(path))
        assert uses_src == (path.name in {"services.py", "serialization.py"}), path


def test_env_example_has_placeholders_only_and_env_is_ignored():
    lines = [l.strip() for l in (ROOT / ".env.example").read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    values = dict(l.split("=", 1) for l in lines)
    assert values["FRONTEND_ORIGIN"] == "http://localhost:3000" and values["ENVIRONMENT"] == "development"
    for key in ("ANGEL_API_KEY", "ANGEL_CLIENT_ID", "ANGEL_PIN", "ANGEL_TOTP_SECRET"):
        assert values[key] == ""
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignore and "!.env.example" in ignore
    assert not (ROOT / ".env").exists()
