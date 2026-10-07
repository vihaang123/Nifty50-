"""POST /api/backtest (the real Phase 7 walk-forward engine, on the synthetic data)."""

import pytest

from src.backtest import run_backtest
from src.universe import get_development_universe

from conftest import strict_json

SUMMARY_KEYS = {"initial_capital", "final_portfolio_value", "cumulative_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"}
BENCHMARK_KEYS = {"final_value", "cumulative_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"}


def post(client, **body):
    return client.post("/api/backtest", json=body)


@pytest.fixture(scope="module")
def full(client):
    """The full default backtest (about 15 seconds), run once through the API."""
    response = post(client, capital=100000, basket_size=10, frequency="quarterly")
    assert response.status_code == 200
    return strict_json(response)


@pytest.fixture(scope="module")
def short(client):
    response = post(client, capital=50000, basket_size=6, frequency="quarterly", start_date="2019-01-01", end_date="2020-12-31")
    assert response.status_code == 200
    return strict_json(response)


# ===================== full run =====================
def test_the_response_has_the_documented_structure(full):
    assert set(full) == {"summary", "benchmark", "equity_curve", "drawdown", "rebalance_history", "rebalances", "skipped_rebalances",
                         "missing_data_events", "data_info", "settings", "is_synthetic", "benchmark_name", "notices"}
    assert set(full["summary"]) == SUMMARY_KEYS and set(full["benchmark"]) == BENCHMARK_KEYS


def test_the_summary_is_the_real_phase_7_result(full):
    s, b = full["summary"], full["benchmark"]
    assert s["initial_capital"] == 100000
    assert s["final_portfolio_value"] == pytest.approx(218256.69, abs=0.01)  # the value the Phase 7 command-line run reported
    assert s["cumulative_return"] == pytest.approx(s["final_portfolio_value"] / 100000 - 1)
    assert s["annualized_volatility"] > 0 and s["max_drawdown"] < 0 and s["sharpe_ratio"] == pytest.approx(s["annualized_return"] / s["annualized_volatility"])
    assert b["final_value"] == pytest.approx(284527.15, abs=0.01) and b["cumulative_return"] == pytest.approx(1.8453, abs=1e-4)
    assert b["sharpe_ratio"] == pytest.approx(b["annualized_return"] / b["annualized_volatility"])


def test_the_equity_curve_is_a_list_of_dated_points(full):
    curve = full["equity_curve"]
    assert len(curve) == 1827 and all(set(p) == {"date", "portfolio_value", "daily_return", "benchmark_value", "benchmark_return"} for p in curve)
    assert curve[0]["date"] == "2019-01-01" and curve[-1]["date"] == "2025-12-31"
    dates = [p["date"] for p in curve]
    assert dates == sorted(dates) and len(set(dates)) == len(dates)
    assert curve[-1]["portfolio_value"] == pytest.approx(full["summary"]["final_portfolio_value"])
    assert curve[-1]["benchmark_value"] == pytest.approx(full["benchmark"]["final_value"])
    assert curve[0]["portfolio_value"] == pytest.approx(100000 * (1 + curve[0]["daily_return"]))


def test_drawdown_series(full):
    table = full["drawdown"]
    assert len(table) == len(full["equity_curve"]) and all(set(p) == {"date", "portfolio_drawdown", "benchmark_drawdown"} for p in table)
    assert min(p["portfolio_drawdown"] for p in table) == pytest.approx(full["summary"]["max_drawdown"])
    assert min(p["benchmark_drawdown"] for p in table) == pytest.approx(full["benchmark"]["max_drawdown"])
    assert all(p["portfolio_drawdown"] <= 0 and p["benchmark_drawdown"] <= 0 for p in table)
    assert [p["date"] for p in table] == [p["date"] for p in full["equity_curve"]]


def test_rebalance_history(full):
    history = full["rebalance_history"]
    assert len(history) == 28 * 10 and {r["rebalance_date"] for r in history} >= {"2019-01-01", "2025-10-01"}
    assert all(set(r) == {"rebalance_date", "symbol", "cap_category", "behavior_class", "weight", "allocation", "reason"} for r in history)
    assert all(r["weight"] == pytest.approx(0.1) for r in history)
    first = [r for r in history if r["rebalance_date"] == "2019-01-01"]
    assert sum(r["allocation"] for r in first) == pytest.approx(100000)


def test_the_rebalance_overview_data_info_and_settings(full):
    done = [r for r in full["rebalances"] if r["status"] == "rebalanced"]
    assert len(done) == 28 and all(r["last_training_date"] < r["rebalance_date"] for r in done)
    assert full["data_info"]["trading_days"] == 2088 and full["data_info"]["n_stocks"] == 10 and full["data_info"]["possible_periods"] == 32
    assert full["settings"]["frequency"] == "quarterly" and full["settings"]["transaction_cost"] == 0 and full["settings"]["slippage"] == 0
    assert full["skipped_rebalances"] == [] and full["missing_data_events"] == []


def test_the_response_says_the_data_and_benchmark_are_synthetic(full):
    assert full["is_synthetic"] is True and full["benchmark_name"] == "Synthetic market index"
    text = " ".join(full["notices"])
    assert "Synthetic pipeline validation only" in text and "Transaction costs = 0" in text


# ===================== the API returns what the engine returns =====================
def test_a_shorter_run_matches_the_engine_called_directly(short, engine):
    direct = run_backtest(engine.stocks, engine.market, get_development_universe(), 50000, 6, "quarterly",
                          start_date="2019-01-01", end_date="2020-12-31")
    assert short["summary"]["final_portfolio_value"] == pytest.approx(direct["summary"]["final_portfolio_value"])
    assert short["summary"]["max_drawdown"] == pytest.approx(direct["summary"]["max_drawdown"])
    assert short["benchmark"]["cumulative_return"] == pytest.approx(direct["summary"]["benchmark_cumulative_return"])
    assert [p["portfolio_value"] for p in short["equity_curve"]] == pytest.approx(direct["equity_curve"]["portfolio_value"].tolist())
    assert [r["symbol"] for r in short["rebalance_history"]] == direct["rebalance_history"]["symbol"].tolist()
    assert short["summary"]["initial_capital"] == 50000 and len(short["rebalance_history"]) == 8 * 6
    assert [r["weight"] for r in short["rebalance_history"]] == pytest.approx(direct["rebalance_history"]["weight"].tolist())
    assert all(r["weight"] == pytest.approx(1 / 6) and r["allocation"] > 0 for r in short["rebalance_history"])  # 6 equal weights, not 0.1


def test_the_period_and_frequency_options_work(client):
    body = post(client, capital=100000, basket_size=6, frequency="semiannual", start_date="2019-01-01", end_date="2020-12-31").json()
    assert body["settings"]["frequency"] == "semiannual"
    assert len([r for r in body["rebalances"] if r["status"] == "rebalanced"]) == 4
    assert body["equity_curve"][0]["date"] == "2019-01-01" and body["equity_curve"][-1]["date"] == "2020-12-31"


def test_the_same_request_twice_gives_the_same_answer(client, short):
    again = post(client, capital=50000, basket_size=6, frequency="quarterly", start_date="2019-01-01", end_date="2020-12-31").json()
    assert again == short


def test_defaults_are_quarterly_100000_and_10(client):
    response = post(client, end_date="2019-06-30")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["initial_capital"] == 100000 and body["settings"]["basket_size"] == 10 and body["settings"]["frequency"] == "quarterly"


# ===================== invalid requests =====================
@pytest.mark.parametrize("body, field", [
    ({"capital": 0}, "capital"), ({"capital": -5}, "capital"), ({"capital": "abc"}, "capital"),
    ({"basket_size": 2}, "basket_size"), ({"basket_size": 4.5}, "basket_size"), ({"basket_size": "ten"}, "basket_size"),
    ({"frequency": "daily"}, "frequency"), ({"frequency": "Quarterly"}, "frequency"), ({"frequency": 3}, "frequency"),
    ({"similarity_threshold": 2}, "similarity_threshold"),
    ({"start_date": "not-a-date"}, "start_date"), ({"end_date": "2020-13-45"}, "end_date"),
])
def test_invalid_parameters_are_a_422_naming_the_field(client, body, field):
    response = post(client, **body)
    assert response.status_code == 422
    assert field in {d["field"] for d in response.json()["error"]["details"]}


def test_start_after_end_is_a_422(client):
    response = post(client, start_date="2021-01-01", end_date="2020-01-01")
    assert response.status_code == 422
    assert "start_date must not be after end_date" in response.text


def test_unknown_fields_are_rejected(client):
    assert post(client, transaction_cost=0.01).status_code == 422  # there is no cost model


def test_a_basket_larger_than_the_available_stocks_is_a_400(client):
    response = post(client, basket_size=11)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_basket_size" and "Invalid basket size" in response.json()["error"]["message"]


def test_a_window_with_no_possible_rebalance_is_a_400_with_a_clear_message(client):
    for window in ({"start_date": "2030-01-01"}, {"start_date": "2018-02-01", "end_date": "2018-03-01"}):
        response = post(client, basket_size=6, **window)
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "invalid_backtest_request" and "No rebalance is possible" in error["message"]
        assert "Traceback" not in response.text


def test_get_is_not_allowed(client):
    assert client.get("/api/backtest").status_code == 405
