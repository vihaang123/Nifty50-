"""GET /api/similarity/{symbol} (the real Phase 5 engine)."""

import pytest

from src.basket import stock_behaviour
from src.similarity import find_similar_stocks
from src.universe import get_development_universe

from conftest import strict_json

SYMBOLS = ["AXISBANK", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "LT", "MARUTI", "RELIANCE", "SBIN", "TCS"]


def test_a_valid_symbol_returns_the_default_five(client):
    response = client.get("/api/similarity/AXISBANK")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol"] == "AXISBANK" and body["top_n"] == 5 and len(body["similar_stocks"]) == 5
    assert all(set(s) == {"rank", "symbol", "similarity", "cap_category", "behavior_class"} for s in body["similar_stocks"])
    assert body["mode"] == "exploratory_full_history"


def test_the_numbers_match_the_phase_5_engine(client, engine):
    body = client.get("/api/similarity/AXISBANK?top_n=5").json()
    expected = find_similar_stocks(engine.profiles, "AXISBANK", 5)
    assert [s["symbol"] for s in body["similar_stocks"]] == expected["symbol"].tolist()
    assert [s["similarity"] for s in body["similar_stocks"]] == pytest.approx(expected["similarity"].tolist())
    assert body["similar_stocks"][0]["symbol"] == "ICICIBANK" and body["similar_stocks"][0]["similarity"] == pytest.approx(0.9737, abs=1e-4)


@pytest.mark.parametrize("top_n", [1, 3, 5, 9])
def test_top_n_controls_the_length(client, top_n):
    body = client.get(f"/api/similarity/TCS?top_n={top_n}").json()
    assert body["top_n"] == top_n and len(body["similar_stocks"]) == top_n


def test_asking_for_more_than_exist_returns_all_the_others(client):
    body = client.get("/api/similarity/TCS?top_n=50").json()
    assert len(body["similar_stocks"]) == 9


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_the_stock_itself_is_never_in_its_own_list(client, symbol):
    body = client.get(f"/api/similarity/{symbol}?top_n=50").json()
    names = [s["symbol"] for s in body["similar_stocks"]]
    assert symbol not in names and sorted(names + [symbol]) == SYMBOLS


def test_ranks_run_from_one_and_similarity_is_descending(client):
    stocks = client.get("/api/similarity/INFY?top_n=9").json()["similar_stocks"]
    assert [s["rank"] for s in stocks] == list(range(1, 10))
    values = [s["similarity"] for s in stocks]
    assert values == sorted(values, reverse=True) and all(-1 <= v <= 1 for v in values)


def test_the_symbol_is_case_insensitive(client):
    assert client.get("/api/similarity/axisbank?top_n=3").json() == client.get("/api/similarity/AXISBANK?top_n=3").json()


def test_an_unknown_symbol_is_a_404_with_a_helpful_message(client):
    response = client.get("/api/similarity/NOPE")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "unknown_symbol" and "Unknown stock symbol 'NOPE'" in error["message"] and "AXISBANK" in error["message"]


@pytest.mark.parametrize("query", ["top_n=0", "top_n=-1", "top_n=51", "top_n=abc", "top_n=2.5"])
def test_bad_top_n_is_a_422(client, query):
    response = client.get(f"/api/similarity/AXISBANK?{query}")
    assert response.status_code == 422 and response.json()["error"]["details"][0]["field"] == "top_n"


def test_similarity_is_symmetric_through_the_api(client):
    ab = {s["symbol"]: s["similarity"] for s in client.get("/api/similarity/AXISBANK?top_n=9").json()["similar_stocks"]}
    ba = {s["symbol"]: s["similarity"] for s in client.get("/api/similarity/ICICIBANK?top_n=9").json()["similar_stocks"]}
    assert ab["ICICIBANK"] == pytest.approx(ba["AXISBANK"])


def test_similarity_response_is_strict_json(client):
    strict_json(client.get("/api/similarity/AXISBANK"))


# ===================== cap category and behaviour class (added for the frontend) =====================
def test_each_stock_carries_its_cap_category_and_behaviour_class(client, engine):
    body = client.get("/api/similarity/AXISBANK?top_n=9").json()
    caps = get_development_universe().set_index("symbol")["cap_category"]
    behaviour = stock_behaviour(engine.lda_output)["behavior_class"]
    assert body["selected"] == {"symbol": "AXISBANK", "cap_category": caps["AXISBANK"], "behavior_class": behaviour["AXISBANK"]}
    for stock in body["similar_stocks"]:
        assert stock["cap_category"] == caps[stock["symbol"]] and stock["behavior_class"] == behaviour[stock["symbol"]]
        assert stock["cap_category"] in {"Large Cap", "Mid Cap", "Small Cap"} and stock["behavior_class"] in {"Defensive", "Balanced", "Aggressive"}


def test_the_new_fields_do_not_change_the_existing_ones(client, engine):
    body = client.get("/api/similarity/AXISBANK?top_n=5").json()
    expected = find_similar_stocks(engine.profiles, "AXISBANK", 5)
    assert [s["symbol"] for s in body["similar_stocks"]] == expected["symbol"].tolist()
    assert [s["rank"] for s in body["similar_stocks"]] == [1, 2, 3, 4, 5]
    assert set(body["similar_stocks"][0]) == {"rank", "symbol", "similarity", "cap_category", "behavior_class"}
