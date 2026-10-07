"""GET /api/dataset and GET /api/universe."""

import shutil

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from api.main import create_app
from src.universe import get_development_universe

from conftest import ROOT


# ===================== dataset =====================
def test_dataset_structure(client):
    response = client.get("/api/dataset")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "provider", "source", "is_synthetic", "start_date", "end_date", "stock_count", "stocks", "trading_days", "observations",
        "market_index", "notice",
    }
    assert isinstance(body["stocks"], list) and all(isinstance(s, str) for s in body["stocks"])


def test_dataset_clearly_says_the_data_is_synthetic(client):
    body = client.get("/api/dataset").json()
    assert body["is_synthetic"] is True and body["source"] == "synthetic"
    assert "synthetic" in body["notice"].lower() and "not real prices" in body["notice"].lower()


def test_dataset_values_come_from_the_actual_data_files(client, engine):
    body = client.get("/api/dataset").json()
    stocks = engine.stocks
    assert body["stock_count"] == 10 == len(body["stocks"]) == stocks["symbol"].nunique()
    assert body["stocks"] == sorted(stocks["symbol"].unique())
    assert body["trading_days"] == 2088 == stocks["date"].nunique()
    assert body["start_date"] == "2018-01-01" == str(stocks["date"].min().date())
    assert body["end_date"] == "2025-12-31" == str(stocks["date"].max().date())
    assert body["market_index"] == "MARKET_INDEX"


def _alternative_config(tmp_path, prices_name, symbols=None, start=None):
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    shutil.copy(ROOT / config["data"]["prices_file"], tmp_path / prices_name)
    shutil.copy(ROOT / config["data"]["market_index_file"], tmp_path / "market.csv")
    config["data"].update(prices_file=prices_name, market_index_file="market.csv", symbols=symbols, start_date=start)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_dataset_follows_the_config_so_nothing_is_hard_coded(tmp_path, monkeypatch):
    path = _alternative_config(tmp_path, "dev_prices_synthetic.csv", symbols=["TCS", "INFY", "ITC"], start="2020-01-01")
    monkeypatch.setenv("CONFIG_PATH", str(path))
    body = TestClient(create_app()).get("/api/dataset").json()
    assert body["stock_count"] == 3 and body["stocks"] == ["INFY", "ITC", "TCS"]
    assert body["start_date"] >= "2020-01-01" and body["trading_days"] < 2088


def test_a_dataset_not_named_synthetic_is_reported_as_not_synthetic(tmp_path, monkeypatch):
    path = _alternative_config(tmp_path, "prices_from_somewhere_else.csv")
    monkeypatch.setenv("CONFIG_PATH", str(path))
    body = TestClient(create_app()).get("/api/dataset").json()
    assert body["is_synthetic"] is False and body["source"] == "local" and "synthetic" not in body["notice"].lower()


# ===================== universe =====================
def test_universe_lists_the_stocks_with_cap_categories(client):
    response = client.get("/api/universe")
    assert response.status_code == 200
    body = response.json()
    assert len(body["stocks"]) == 10
    assert all(set(s) == {"symbol", "cap_category"} for s in body["stocks"])
    assert {s["cap_category"] for s in body["stocks"]} == {"Large Cap", "Mid Cap", "Small Cap"}
    assert body["counts"] == {"Large Cap": 4, "Mid Cap": 3, "Small Cap": 3}


def test_universe_is_the_existing_universe_module_not_a_copy(client):
    body = client.get("/api/universe").json()
    expected = get_development_universe()
    assert pd.DataFrame(body["stocks"]).equals(expected)
    assert body["stocks"][0] == {"symbol": "AXISBANK", "cap_category": "Large Cap"}


def test_universe_says_the_cap_categories_are_synthetic(client):
    body = client.get("/api/universe").json()
    assert body["is_synthetic"] is True
    assert body["notice"] == "Synthetic development cap classifications used only for testing and demonstration."


def test_universe_and_dataset_agree_on_the_symbols(client):
    universe = {s["symbol"] for s in client.get("/api/universe").json()["stocks"]}
    assert universe == set(client.get("/api/dataset").json()["stocks"])
