"""POST /api/basket/generate (the real Phase 6 generator, on the real PCA/LDA/similarity of the synthetic data)."""

import pytest

from src.basket import basket_statistics, generate_basket
from src.labels import CLASS_NAMES
from src.universe import get_development_universe

from conftest import strict_json


def post(client, **body):
    return client.post("/api/basket/generate", json=body)


@pytest.fixture(scope="module")
def basket(client):
    return post(client, capital=100000, basket_size=10, similarity_threshold=0.90).json()


# ===================== valid requests =====================
def test_a_valid_request_returns_the_documented_structure(basket):
    assert set(basket) == {"capital", "basket_size", "similarity_threshold", "total_weight", "total_allocation", "stocks", "statistics",
                           "target_allocation", "planned_allocation", "notes", "n_eligible", "is_synthetic", "mode", "notice"}
    assert all(set(s) == {"symbol", "cap_category", "behavior_class", "weight", "allocation", "reason"} for s in basket["stocks"])
    assert set(basket["statistics"]) == {"behavior_distribution", "cap_distribution", "average_similarity", "maximum_similarity",
                                         "maximum_pair", "pairs_above_threshold", "n_pairs"}
    assert basket["capital"] == 100000 and basket["basket_size"] == 10 and basket["similarity_threshold"] == 0.9


def test_allocation_totals(basket):
    assert basket["total_weight"] == 1.0 and basket["total_allocation"] == 100000.0
    assert sum(s["weight"] for s in basket["stocks"]) == pytest.approx(1.0)
    assert sum(s["allocation"] for s in basket["stocks"]) == pytest.approx(100000)
    assert all(s["weight"] == pytest.approx(0.1) and s["allocation"] == pytest.approx(10000) for s in basket["stocks"])  # equal weights


@pytest.mark.parametrize("size, capital", [(3, 30000), (6, 50000), (7, 123456.78), (10, 1)])
def test_totals_and_counts_for_other_sizes_and_capitals(client, size, capital):
    body = post(client, capital=capital, basket_size=size).json()
    assert len(body["stocks"]) == size == body["basket_size"]
    assert body["total_allocation"] == pytest.approx(capital) and body["total_weight"] == 1.0
    assert len({s["symbol"] for s in body["stocks"]}) == size  # no stock twice


def test_the_response_is_exactly_what_the_engine_produces(client, engine):
    body = post(client, capital=50000, basket_size=6, similarity_threshold=0.9).json()
    direct = generate_basket(engine.lda_output, engine.matrix, get_development_universe(), 6, 50000, 0.9)
    assert [s["symbol"] for s in body["stocks"]] == direct.basket["symbol"].tolist()
    assert [s["reason"] for s in body["stocks"]] == direct.basket["reason"].tolist()
    assert [s["behavior_class"] for s in body["stocks"]] == direct.basket["behavior_class"].tolist()
    assert [s["allocation"] for s in body["stocks"]] == pytest.approx(direct.basket["allocation"].tolist())
    assert body["notes"] == direct.notes and body["target_allocation"] == direct.target_allocation
    stats = basket_statistics(direct.basket, engine.matrix, 0.9)
    assert body["statistics"]["average_similarity"] == pytest.approx(stats["average_similarity"])
    assert body["statistics"]["maximum_similarity"] == pytest.approx(stats["maximum_similarity"])
    assert body["statistics"]["maximum_pair"] == list(stats["maximum_pair"])


def test_the_multi_cap_split_is_respected(client):
    body = post(client, capital=50000, basket_size=6).json()
    assert body["statistics"]["cap_distribution"] == {
        "Large Cap": {"count": 2, "percentage": pytest.approx(100 / 3)},
        "Mid Cap": {"count": 2, "percentage": pytest.approx(100 / 3)},
        "Small Cap": {"count": 2, "percentage": pytest.approx(100 / 3)},
    }
    ten = post(client, basket_size=10).json()["statistics"]["cap_distribution"]
    assert [ten[c]["count"] for c in ("Large Cap", "Mid Cap", "Small Cap")] == [4, 3, 3]


def test_statistics_are_consistent(basket):
    stats = basket["statistics"]
    assert list(stats["behavior_distribution"]) == CLASS_NAMES
    assert sum(d["count"] for d in stats["behavior_distribution"].values()) == 10
    assert sum(d["percentage"] for d in stats["behavior_distribution"].values()) == pytest.approx(100)
    assert sum(d["count"] for d in stats["cap_distribution"].values()) == 10
    assert -1 <= stats["average_similarity"] <= stats["maximum_similarity"] <= 1
    assert stats["n_pairs"] == 45 and len(stats["maximum_pair"]) == 2 and set(stats["maximum_pair"]) <= {s["symbol"] for s in basket["stocks"]}


def test_every_stock_has_a_reason_and_the_response_is_labelled(basket):
    assert all(len(s["reason"]) > 20 for s in basket["stocks"])
    assert basket["is_synthetic"] is True and basket["mode"] == "exploratory_full_history"
    assert "not an investment recommendation" in basket["notice"] and "Synthetic development cap classifications" in basket["notice"]


def test_the_same_request_gives_the_same_basket(client, basket):
    assert post(client, capital=100000, basket_size=10, similarity_threshold=0.90).json() == basket


def test_an_empty_body_uses_the_documented_defaults(client):
    body = client.post("/api/basket/generate", json={}).json()
    assert body["capital"] == 100000 and body["basket_size"] == 10 and body["similarity_threshold"] == 0.9


def test_an_integer_capital_and_a_threshold_at_the_limits_are_accepted(client):
    assert post(client, capital=75000, basket_size=4, similarity_threshold=1).status_code == 200
    assert post(client, capital=75000, basket_size=4, similarity_threshold=-1).status_code == 200


def test_the_response_is_strict_json(client):
    strict_json(post(client, capital=100000, basket_size=8))


# ===================== invalid requests =====================
@pytest.mark.parametrize("capital", [0, -1, -100000, "abc", None, [100]])
def test_invalid_capital_is_a_422(client, capital):
    response = post(client, capital=capital, basket_size=10)
    assert response.status_code == 422
    assert any(d["field"] == "capital" for d in response.json()["error"]["details"])


@pytest.mark.parametrize("size", [0, 1, 2, -3, 3.5, "ten", None, True])
def test_a_basket_size_below_three_or_not_a_whole_number_is_a_422(client, size):
    response = post(client, capital=100000, basket_size=size)
    assert response.status_code == 422
    assert any(d["field"] == "basket_size" for d in response.json()["error"]["details"])


def test_a_basket_larger_than_the_available_stocks_is_a_400(client):
    response = post(client, capital=100000, basket_size=11)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_basket_size" and "Invalid basket size" in error["message"] and "10 stocks" in error["message"]
    assert post(client, capital=100000, basket_size=500).status_code == 400
    assert post(client, capital=100000, basket_size=501).status_code == 422  # beyond the schema limit


@pytest.mark.parametrize("threshold", [1.5, -1.01, 2, "high", None])
def test_invalid_similarity_threshold_is_a_422(client, threshold):
    response = post(client, capital=100000, basket_size=10, similarity_threshold=threshold)
    assert response.status_code == 422
    assert any(d["field"] == "similarity_threshold" for d in response.json()["error"]["details"])


def test_non_finite_numbers_are_rejected(client):
    for raw in ('{"capital": NaN}', '{"capital": Infinity}', '{"similarity_threshold": NaN}'):
        response = client.post("/api/basket/generate", content=raw, headers={"Content-Type": "application/json"})
        assert response.status_code == 422, raw


def test_unknown_fields_missing_body_and_bad_json_are_422(client):
    assert post(client, capital=100000, basket_size=10, risk_appetite="high").status_code == 422
    assert client.post("/api/basket/generate").status_code == 422
    assert client.post("/api/basket/generate", content="{not json", headers={"Content-Type": "application/json"}).status_code == 422
    assert client.post("/api/basket/generate", json=[1, 2, 3]).status_code == 422


def test_get_is_not_allowed(client):
    assert client.get("/api/basket/generate").status_code == 405


# ===================== the threshold really reaches the engine =====================
@pytest.mark.parametrize("threshold", [0.5, 0.95, 0.99])
def test_the_similarity_threshold_is_passed_to_the_generator_and_the_statistics(client, engine, threshold):
    body = post(client, capital=100000, basket_size=8, similarity_threshold=threshold).json()
    direct = generate_basket(engine.lda_output, engine.matrix, get_development_universe(), 8, 100000, threshold)
    stats = basket_statistics(direct.basket, engine.matrix, threshold)
    assert body["similarity_threshold"] == threshold
    assert [s["symbol"] for s in body["stocks"]] == direct.basket["symbol"].tolist()
    assert body["statistics"]["pairs_above_threshold"] == stats["pairs_above_threshold"]
    assert body["notes"] == direct.notes


def test_a_lower_threshold_flags_more_pairs_as_too_similar(client):
    strict = post(client, basket_size=8, similarity_threshold=0.5).json()["statistics"]["pairs_above_threshold"]
    loose = post(client, basket_size=8, similarity_threshold=0.99).json()["statistics"]["pairs_above_threshold"]
    assert strict > loose
