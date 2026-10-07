"""
Tests for Phase 7 (src/backtest.py): the walk-forward backtest.

Groups:
  A. metric and return arithmetic, checked against hand-computed numbers
  B. rebalance dates and backtest mechanics
  C. LOOK-AHEAD tests: change the future and prove nothing earlier moves
  D. missing prices, errors, outputs, charts, command line, architecture

Run:  pytest -q
"""

import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import src.backtest as bt
from src.backtest import (
    DEFAULT_MIN_HISTORY_DAYS,
    HISTORY_COLUMNS,
    RESULT_COLUMNS,
    SUMMARY_KEYS,
    calculate_benchmark,
    calculate_drawdown,
    calculate_metrics,
    calculate_portfolio_returns,
    format_backtest_report,
    generate_rebalance_dates,
    main,
    plot_drawdown,
    plot_equity_curve,
    result_to_jsonable,
    run_backtest,
    run_single_rebalance,
    save_backtest_outputs,
)
from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import MIN_HISTORY_ROWS
from src.pca_model import fit_pca
from src.universe import get_development_universe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE = get_development_universe()


# ======================= shared data =======================
@pytest.fixture(scope="module")
def dev():
    config = load_config(PROJECT_ROOT / "config.yaml")
    stocks = get_provider(config, PROJECT_ROOT).get_historical_data()
    market = LocalDataProvider(PROJECT_ROOT / config["data"]["market_index_file"]).get_historical_data()
    return stocks, market


def future_changed(stocks: pd.DataFrame, market: pd.DataFrame, cutoff: str, seed: int = 7):
    """Copies of the data that are IDENTICAL before `cutoff` and wildly different from `cutoff` on."""
    rng = np.random.default_rng(seed)
    cut = pd.Timestamp(cutoff)

    def change(table):
        table = table.copy()
        later = (table["date"] >= cut).to_numpy()
        factor = rng.uniform(0.6, 1.6, size=len(table))
        for column in ("open", "high", "low", "close"):
            table.loc[later, column] = table.loc[later, column].to_numpy() * factor[later]
        table.loc[later, "volume"] = table.loc[later, "volume"].to_numpy() * rng.uniform(0.5, 3.0, size=later.sum())
        return table

    return change(stocks), change(market)


@pytest.fixture(scope="module")
def short_run(dev):
    """Six-stock backtest over 2019-2020 (8 quarterly rebalances), run once and shared."""
    stocks, market = dev
    return run_backtest(stocks, market, UNIVERSE, capital=100_000, basket_size=6, end_date="2020-12-31")


# ---- a small fast world for mechanics: 4 stocks, ~300 trading days, with a stand-in for the heavy rebalance ----
@pytest.fixture
def small_world():
    dates = pd.bdate_range("2019-01-01", periods=320)
    rng = np.random.default_rng(3)
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    rows = []
    for s in symbols:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(dates)))
        rows.append(pd.DataFrame({"date": dates, "symbol": s, "open": close, "high": close, "low": close, "close": close, "volume": 1000.0}))
    stocks = pd.concat(rows, ignore_index=True)
    index = 1000 * np.cumprod(1 + rng.normal(0.0004, 0.008, len(dates)))
    market = pd.DataFrame({"date": dates, "symbol": "IDX", "open": index, "high": index, "low": index, "close": index, "volume": 0.0})
    universe = pd.DataFrame({"symbol": symbols, "cap_category": ["Large Cap", "Large Cap", "Mid Cap", "Small Cap"]})
    return stocks, market, universe


def fake_rebalance(symbols_by_call):
    """A stand-in for run_single_rebalance that returns a fixed basket (or raises), so mechanics run fast."""
    calls = []

    def fake(stock_data, market_data, rebalance_date, universe, basket_size, capital, *args, **kwargs):
        calls.append(pd.Timestamp(rebalance_date))
        choice = symbols_by_call[min(len(calls) - 1, len(symbols_by_call) - 1)]
        if choice is None:
            raise ValueError("pretend the models could not be fitted")
        n = len(choice)
        basket = pd.DataFrame(
            {"symbol": choice, "cap_category": "Large Cap", "behavior_class": "Balanced", "weight": 1 / n,
             "allocation": capital / n, "reason": "test"}
        )
        return {"rebalance_date": pd.Timestamp(rebalance_date), "last_training_date": pd.Timestamp(rebalance_date) - pd.Timedelta(days=1),
                "n_training_rows": 1, "basket_result": SimpleNamespace(basket=basket, notes=[])}

    return fake, calls


# ===========================================================================================
# A. ARITHMETIC
# ===========================================================================================
# ---- test 1: daily portfolio return ----
def test_daily_portfolio_return_is_the_mean_of_the_held_stocks_returns():
    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0, 99.0], "B": [50.0, 50.0, 55.0, 60.5]}, index=dates)
    out = calculate_portfolio_returns(prices, ["A", "B"], dates[1:])
    # day 2: A +10%, B 0%  -> +5%;  day 3: A -10%, B +10% -> 0%;  day 4: A 0%, B +10% -> +5%
    assert out["daily_return"].to_numpy() == pytest.approx([0.05, 0.0, 0.05])
    assert out["n_active"].tolist() == [2, 2, 2] and out["stopped"] == []


# ---- test 2: equal weights ----
def test_equal_weight_return_with_three_stocks():
    dates = pd.bdate_range("2024-01-01", periods=2)
    prices = pd.DataFrame({"A": [100.0, 103.0], "B": [100.0, 96.0], "C": [100.0, 107.0]}, index=dates)
    out = calculate_portfolio_returns(prices, ["A", "B", "C"], dates[1:])
    assert out["daily_return"].iloc[0] == pytest.approx((0.03 - 0.04 + 0.07) / 3)


def test_the_first_day_of_a_period_uses_the_close_before_the_period():
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [100.0, 100.0, 120.0, 120.0, 120.0]}, index=dates)
    out = calculate_portfolio_returns(prices, ["A"], dates[2:])  # the period starts on day 3
    assert out["daily_return"].iloc[0] == pytest.approx(0.20)  # 120 / 100 - 1, the gap from the previous close


# ---- test 3: cumulative return ----
def test_cumulative_return_is_final_over_initial_minus_one():
    returns = [0.10, -0.05, 0.02]
    metrics = calculate_metrics(returns, years=1.0)
    assert metrics["cumulative_return"] == pytest.approx(1.10 * 0.95 * 1.02 - 1)
    assert 100_000 * (1 + metrics["cumulative_return"]) == pytest.approx(100_000 * 1.10 * 0.95 * 1.02)


# ---- test 4: annualised return uses the real duration ----
def test_annualised_return_uses_the_actual_duration():
    returns = [0.21] + [0.0] * 4  # +21% in total
    assert calculate_metrics(returns, years=2.0)["annualized_return"] == pytest.approx(0.10)  # 1.21 ** 0.5 - 1
    assert calculate_metrics(returns, years=1.0)["annualized_return"] == pytest.approx(0.21)
    assert calculate_metrics(returns, years=0.5)["annualized_return"] == pytest.approx(1.21**2 - 1)


# ---- test 5: annualised volatility ----
def test_annualised_volatility_is_daily_std_times_root_252():
    returns = [0.01, -0.01, 0.02, 0.0]
    expected = np.std(returns, ddof=1) * math.sqrt(252)
    assert calculate_metrics(returns, years=1.0)["annualized_volatility"] == pytest.approx(expected)


# ---- test 6: Sharpe ----
def test_sharpe_is_annualised_return_over_annualised_volatility_with_zero_risk_free_rate():
    returns = [0.01, -0.005, 0.015, 0.002]
    m = calculate_metrics(returns, years=0.75)
    assert m["sharpe_ratio"] == pytest.approx(m["annualized_return"] / m["annualized_volatility"])


def test_sharpe_is_none_when_there_is_no_volatility():
    assert calculate_metrics([0.001, 0.001, 0.001], years=1.0)["sharpe_ratio"] is None


# ---- test 7: max drawdown ----
def test_max_drawdown_is_the_worst_fall_from_a_running_peak():
    # values: 100 -> 120 -> 90 -> 110 -> 80   (start value 1.0 is the first peak)
    returns = [0.20, -0.25, 110 / 90 - 1, 80 / 110 - 1]
    assert calculate_metrics(returns, years=1.0)["max_drawdown"] == pytest.approx(80 / 120 - 1)


def test_drawdown_series_values():
    out = calculate_drawdown([100, 120, 90, 110, 80, 130])
    assert out.to_numpy() == pytest.approx([0, 0, -0.25, -10 / 120, -1 / 3, 0])


def test_the_starting_value_counts_as_a_peak_so_an_immediate_loss_is_a_drawdown():
    assert calculate_metrics([-0.10, 0.01], years=1.0)["max_drawdown"] == pytest.approx(-0.10)


@pytest.mark.parametrize("bad", [[0.01], [0.01, float("nan")]])
def test_metrics_reject_too_little_or_missing_data(bad):
    with pytest.raises(ValueError):
        calculate_metrics(bad, years=1.0)


@pytest.mark.parametrize("years", [0, -1, float("nan"), float("inf")])
def test_metrics_reject_a_bad_duration(years):
    with pytest.raises(ValueError, match="years"):
        calculate_metrics([0.01, 0.02], years=years)


# ---- test 8: benchmark ----
def test_benchmark_return_and_value_follow_the_market_index():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([1000.0, 1010.0, 1000.0, 1100.0], index=dates)
    out = calculate_benchmark(close, dates[1:], 50_000)
    assert out["benchmark_return"].to_numpy() == pytest.approx([0.01, 1000 / 1010 - 1, 0.10])
    assert out["benchmark_value"].to_numpy() == pytest.approx([50_500, 50_000, 55_000])


def test_benchmark_with_a_missing_index_price_is_an_error_not_a_guess():
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([1000.0, np.nan, 1000.0, 1100.0], index=dates)
    with pytest.raises(ValueError, match="missing price"):
        calculate_benchmark(close, dates[1:], 50_000)
    with pytest.raises(ValueError, match="no close"):
        calculate_benchmark(close.iloc[:3], dates[1:], 50_000)


# ===========================================================================================
# B. REBALANCE DATES AND MECHANICS
# ===========================================================================================
# ---- test 9: quarterly dates ----
def test_quarterly_dates_are_the_first_trading_day_of_each_quarter():
    dates = pd.bdate_range("2019-01-01", "2020-12-31")
    out = generate_rebalance_dates(dates, "quarterly", min_history_days=100)
    assert out[0] == pd.Timestamp("2019-07-01")  # 2019-04-01 is only ~65 trading days in
    assert out == [pd.Timestamp(d) for d in ["2019-07-01", "2019-10-01", "2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]]
    assert all(d in set(dates) for d in out)


def test_the_first_trading_day_is_used_even_when_the_quarter_starts_on_a_weekend():
    dates = pd.bdate_range("2019-01-01", "2020-12-31")
    out = generate_rebalance_dates(dates, "quarterly", min_history_days=61)
    assert pd.Timestamp("2019-09-30") not in out and pd.Timestamp("2019-07-01") in out  # a Monday
    assert pd.Timestamp("2019-04-01") in out  # 2019-04-01 is a Monday and has 65 days of history


def test_dates_come_from_the_dataset_not_the_calendar():
    dates = pd.bdate_range("2019-01-01", "2020-12-31").drop(pd.Timestamp("2020-01-01"))  # a holiday
    out = generate_rebalance_dates(dates, "quarterly", min_history_days=100)
    assert pd.Timestamp("2020-01-02") in out and pd.Timestamp("2020-01-01") not in out


def test_a_quarter_whose_first_day_is_too_early_is_skipped_not_moved_later():
    dates = pd.bdate_range("2019-01-01", "2019-12-31")
    out = generate_rebalance_dates(dates, "quarterly", min_history_days=70)
    assert pd.Timestamp("2019-04-01") not in out and pd.Timestamp("2019-04-02") not in out
    assert out[0] == pd.Timestamp("2019-07-01")


@pytest.mark.parametrize(
    "frequency, expected_count",
    [("monthly", 24), ("quarterly", 8), ("semiannual", 4), ("annual", 2)],
)
def test_other_frequencies(frequency, expected_count):
    dates = pd.bdate_range("2019-01-01", "2020-12-31")
    out = generate_rebalance_dates(dates, frequency, min_history_days=61)
    assert len(out) <= expected_count and len(out) >= expected_count - 3
    assert out == sorted(out) and len(set(out)) == len(out)


def test_start_and_end_dates_restrict_the_rebalance_dates():
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    out = generate_rebalance_dates(dates, "quarterly", 100, start_date="2020-01-01", end_date="2020-12-31")
    assert out[0] >= pd.Timestamp("2020-01-01") and out[-1] <= pd.Timestamp("2020-12-31") and len(out) == 4
    with pytest.raises(ValueError, match="after end_date"):
        generate_rebalance_dates(dates, "quarterly", 100, start_date="2021-01-01", end_date="2020-01-01")


# ---- test 13: insufficient history ----
def test_insufficient_history_prevents_a_rebalance(dev):
    stocks, market = dev
    short = pd.bdate_range("2019-01-01", periods=80)
    assert generate_rebalance_dates(short, "quarterly", min_history_days=252) == []
    with pytest.raises(ValueError, match="Not enough history"):
        run_single_rebalance(stocks, market, "2018-06-01", UNIVERSE)  # only ~108 trading days before it
    with pytest.raises(ValueError, match="No rebalance is possible"):
        run_backtest(stocks[stocks["date"] < "2018-09-01"], market, UNIVERSE, basket_size=6)


def test_exactly_the_required_history_is_enough_and_one_day_less_is_not():
    dates = pd.bdate_range("2019-01-01", "2020-12-31")
    april = pd.Timestamp("2019-04-01")
    assert dates.get_loc(april) == 64  # 64 trading days come before it
    assert april in generate_rebalance_dates(dates, "quarterly", min_history_days=64)
    assert april not in generate_rebalance_dates(dates, "quarterly", min_history_days=65)


def test_a_single_rebalance_needs_exactly_the_required_training_days(dev):
    stocks, market = dev
    days = sorted(stocks["date"].unique())
    run_single_rebalance(stocks, market, days[252], UNIVERSE, basket_size=6, min_history_days=252)  # 252 days before it: enough
    with pytest.raises(ValueError, match="Not enough history .*251 trading days, at least 252"):
        run_single_rebalance(stocks, market, days[251], UNIVERSE, basket_size=6, min_history_days=252)


@pytest.mark.parametrize("bad", [60, 10, 0, -5, 100.5, True])
def test_min_history_must_exceed_the_feature_warm_up(bad):
    with pytest.raises(ValueError, match="min_history_days"):
        generate_rebalance_dates(pd.bdate_range("2019-01-01", periods=400), "quarterly", min_history_days=bad)
    assert MIN_HISTORY_ROWS == 60


def test_the_default_first_rebalance_waits_for_a_full_year_of_history(short_run):
    first = short_run["data_info"]["first_rebalance"]
    assert first == pd.Timestamp("2019-01-01")
    history = short_run["data_info"]
    assert history["start_date"] == pd.Timestamp("2018-01-01") and DEFAULT_MIN_HISTORY_DAYS == 252


# ---- test 10: portfolio value ----
def test_portfolio_value_compounds_from_the_initial_capital(short_run):
    eq = short_run["equity_curve"]
    assert list(eq.columns) == RESULT_COLUMNS
    assert eq["portfolio_value"].to_numpy() == pytest.approx(100_000 * (1 + eq["daily_return"]).cumprod().to_numpy())
    assert eq["portfolio_value"].iloc[0] == pytest.approx(100_000 * (1 + eq["daily_return"].iloc[0]))
    assert short_run["summary"]["final_portfolio_value"] == pytest.approx(eq["portfolio_value"].iloc[-1])
    assert short_run["summary"]["initial_capital"] == 100_000


def test_benchmark_columns_follow_the_market_index_on_the_same_dates(short_run, dev):
    _, market = dev
    eq = short_run["equity_curve"]
    close = market.set_index("date")["close"]
    expected = (close / close.shift(1) - 1).loc[eq["date"]].to_numpy()
    assert eq["benchmark_return"].to_numpy() == pytest.approx(expected)
    assert eq["benchmark_value"].to_numpy() == pytest.approx(100_000 * np.cumprod(1 + expected))


def test_the_curve_covers_every_trading_day_from_the_first_rebalance_exactly_once(short_run, dev):
    stocks, _ = dev
    eq = short_run["equity_curve"]
    expected = sorted(d for d in stocks["date"].unique() if pd.Timestamp("2019-01-01") <= d <= pd.Timestamp("2020-12-31"))
    assert eq["date"].tolist() == expected and eq["date"].is_unique


# ---- test 11: weights stay equal; the return really is the mean of the held stocks' returns ----
def test_weights_remain_equal_and_the_daily_return_matches_an_independent_calculation(short_run, dev):
    stocks, _ = dev
    history = short_run["rebalance_history"]
    assert list(history.columns) == HISTORY_COLUMNS
    for date, group in history.groupby("rebalance_date"):
        assert len(group) == 6 and group["weight"].to_numpy() == pytest.approx(1 / 6)
        assert group["allocation"].sum() == pytest.approx(short_run["rebalances"][[r["rebalance_date"] for r in short_run["rebalances"]].index(date)]["capital"])
    wide = stocks.pivot(index="date", columns="symbol", values="close")
    returns = wide / wide.shift(1) - 1
    eq = short_run["equity_curve"].set_index("date")
    dates = sorted(history["rebalance_date"].unique())
    for i, start in enumerate(dates):
        stop = dates[i + 1] if i + 1 < len(dates) else eq.index[-1] + pd.Timedelta(days=1)
        held = history[history["rebalance_date"] == start]["symbol"].tolist()
        window = eq.loc[(eq.index >= start) & (eq.index < stop)]
        expected = returns.loc[window.index, held].mean(axis=1)
        assert window["daily_return"].to_numpy() == pytest.approx(expected.to_numpy())


# ---- test 12: zero costs ----
def test_there_are_no_transaction_costs_or_slippage(short_run):
    assert short_run["settings"]["transaction_cost"] == 0 and short_run["settings"]["slippage"] == 0
    assert short_run["settings"]["risk_free_rate"] == 0
    import inspect

    names = set(inspect.signature(run_backtest).parameters)
    assert not any("cost" in n or "slippage" in n or "fee" in n for n in names)
    eq = short_run["equity_curve"]
    # a frictionless equity curve is exactly the compounded daily returns (nothing is deducted at a rebalance)
    assert eq["portfolio_value"].iloc[-1] == pytest.approx(100_000 * float(np.prod(1 + eq["daily_return"])))


# ---- test 14: determinism ----
def test_the_same_backtest_twice_gives_identical_results(dev, short_run):
    stocks, market = dev
    again = run_backtest(stocks, market, UNIVERSE, capital=100_000, basket_size=6, end_date="2020-12-31")
    pd.testing.assert_frame_equal(short_run["equity_curve"], again["equity_curve"])
    pd.testing.assert_frame_equal(short_run["rebalance_history"], again["rebalance_history"])
    pd.testing.assert_frame_equal(short_run["drawdown"], again["drawdown"])
    assert short_run["summary"] == again["summary"]


def test_summary_matches_a_recomputation_from_the_equity_curve(short_run):
    eq, s, info = short_run["equity_curve"], short_run["summary"], short_run["data_info"]
    again = calculate_metrics(eq["daily_return"], info["years"])
    bench = calculate_metrics(eq["benchmark_return"], info["years"])
    for key, value in again.items():
        assert s[key] == pytest.approx(value)
    assert s["benchmark_cumulative_return"] == pytest.approx(bench["cumulative_return"])
    assert s["benchmark_sharpe_ratio"] == pytest.approx(bench["sharpe_ratio"])
    assert s["benchmark_max_drawdown"] == pytest.approx(bench["max_drawdown"])
    assert s["benchmark_volatility"] == pytest.approx(bench["annualized_volatility"])
    assert s["benchmark_annualized_return"] == pytest.approx(bench["annualized_return"])
    assert s["cumulative_return"] == pytest.approx(eq["portfolio_value"].iloc[-1] / 100_000 - 1)
    # the duration is the real calendar span from the close before the first return to the last date
    assert info["years"] == pytest.approx((info["backtest_end"] - info["baseline_date"]).days / 365.25)
    assert info["baseline_date"] < info["backtest_start"]


def test_the_drawdown_table_agrees_with_the_summary(short_run):
    table, s = short_run["drawdown"], short_run["summary"]
    assert list(table.columns) == ["date", "portfolio_drawdown", "benchmark_drawdown"]
    assert table["portfolio_drawdown"].min() == pytest.approx(s["max_drawdown"])
    assert table["benchmark_drawdown"].min() == pytest.approx(s["benchmark_max_drawdown"])
    assert (table["portfolio_drawdown"] <= 1e-12).all()


def test_the_result_is_structured_data_with_no_printing(short_run):
    assert set(short_run) >= {"summary", "equity_curve", "drawdown", "rebalance_history", "rebalances", "skipped_rebalances", "data_info", "settings"}
    assert list(short_run["summary"]) == SUMMARY_KEYS
    assert short_run["data_info"]["n_stocks"] == 10 and short_run["data_info"]["trading_days"] == 2088
    assert len([r for r in short_run["rebalances"] if r["status"] == "rebalanced"]) == 8


def test_the_inputs_are_not_modified(dev):
    stocks, market = dev
    before_s, before_m = stocks.copy(deep=True), market.copy(deep=True)
    run_single_rebalance(stocks, market, "2019-04-01", UNIVERSE, basket_size=6)
    pd.testing.assert_frame_equal(stocks, before_s)
    pd.testing.assert_frame_equal(market, before_m)


# ---- mechanics with a stand-in rebalance (fast) ----
def test_a_failed_rebalance_keeps_the_previous_basket_and_is_reported(small_world, monkeypatch):
    stocks, market, universe = small_world
    fake, calls = fake_rebalance([["AAA", "BBB"], None, ["CCC", "DDD"]])
    monkeypatch.setattr(bt, "run_single_rebalance", fake)
    result = run_backtest(stocks, market, universe, basket_size=3, min_history_days=70)
    statuses = [r["status"] for r in result["rebalances"]]
    assert statuses[:3] == ["rebalanced", "skipped_kept_previous_basket", "rebalanced"]
    assert len(result["skipped_rebalances"]) == 1 and "could not be fitted" in result["skipped_rebalances"][0]["reason"]
    assert result["rebalances"][1]["symbols"] == ["AAA", "BBB"]  # still holding the old basket
    assert result["rebalance_history"]["rebalance_date"].nunique() == len(statuses) - 1


def test_days_before_the_first_successful_rebalance_are_not_part_of_the_backtest(small_world, monkeypatch):
    stocks, market, universe = small_world
    fake, calls = fake_rebalance([None, ["AAA", "BBB"]])
    monkeypatch.setattr(bt, "run_single_rebalance", fake)
    result = run_backtest(stocks, market, universe, basket_size=3, min_history_days=70)
    assert result["rebalances"][0]["status"] == "skipped_no_basket"
    assert result["equity_curve"]["date"].iloc[0] == calls[1]  # starts at the first rebalance that worked


def test_capital_at_each_rebalance_is_the_portfolio_value_then(small_world, monkeypatch):
    stocks, market, universe = small_world
    fake, _ = fake_rebalance([["AAA", "BBB"]])
    monkeypatch.setattr(bt, "run_single_rebalance", fake)
    result = run_backtest(stocks, market, universe, capital=100_000, basket_size=3, min_history_days=70)
    done = [r for r in result["rebalances"] if r["status"] == "rebalanced"]
    assert done[0]["capital"] == 100_000
    eq = result["equity_curve"].set_index("date")
    for record in done[1:]:
        before = eq.loc[eq.index < record["rebalance_date"], "portfolio_value"].iloc[-1]
        assert record["capital"] == pytest.approx(before)


# ===========================================================================================
# C. LOOK-AHEAD TESTS  (change the future; nothing known earlier may move)
# ===========================================================================================
CUT = "2019-10-01"


@pytest.fixture(scope="module")
def twin(dev):
    """Rebalance at CUT on data A and on data B that differs only from CUT onwards."""
    stocks, market = dev
    stocks_b, market_b = future_changed(stocks, market, CUT)
    a = run_single_rebalance(stocks, market, CUT, UNIVERSE, basket_size=6)
    b = run_single_rebalance(stocks_b, market_b, CUT, UNIVERSE, basket_size=6)
    return SimpleNamespace(a=a, b=b, stocks=stocks, market=market, stocks_b=stocks_b, market_b=market_b)


def test_the_two_datasets_really_differ_only_in_the_future(twin):
    before = twin.stocks["date"] < CUT
    pd.testing.assert_frame_equal(twin.stocks[before], twin.stocks_b[before])
    assert not np.allclose(twin.stocks.loc[~before, "close"], twin.stocks_b.loc[~before, "close"])
    assert not np.allclose(twin.market.loc[twin.market["date"] >= CUT, "close"], twin.market_b.loc[twin.market_b["date"] >= CUT, "close"])


# ---- test 15: future prices cannot change an earlier basket ----
def test_t15_future_prices_cannot_change_the_basket(twin):
    pd.testing.assert_frame_equal(twin.a["basket_result"].basket, twin.b["basket_result"].basket)
    assert twin.a["basket_result"].notes == twin.b["basket_result"].notes


def test_t15_a_whole_backtest_is_unchanged_before_the_change_and_changes_after_it(dev):
    stocks, market = dev
    stocks_b, market_b = future_changed(stocks, market, CUT)
    a = run_backtest(stocks, market, UNIVERSE, basket_size=6, end_date="2020-03-31")
    b = run_backtest(stocks_b, market_b, UNIVERSE, basket_size=6, end_date="2020-03-31")
    cut = pd.Timestamp(CUT)
    early_a = a["rebalance_history"][a["rebalance_history"]["rebalance_date"] <= cut]
    early_b = b["rebalance_history"][b["rebalance_history"]["rebalance_date"] <= cut]
    assert len(early_a) == 6 * 4 and early_a.equals(early_b)  # rebalances at 2019-01, 04, 07 and 10-01
    ea, eb = a["equity_curve"], b["equity_curve"]
    before = ea["date"] < cut
    pd.testing.assert_frame_equal(ea[before], eb[before])
    assert not np.allclose(ea.loc[~before, "portfolio_value"], eb.loc[~before, "portfolio_value"])  # the change is real


# ---- test 16: PCA ----
def test_t16_future_data_cannot_change_pca(twin):
    a, b = twin.a["pca_model"], twin.b["pca_model"]
    np.testing.assert_array_equal(a.pca.components_, b.pca.components_)
    np.testing.assert_array_equal(a.pca.explained_variance_ratio_, b.pca.explained_variance_ratio_)
    np.testing.assert_array_equal(a.scaler.mean_, b.scaler.mean_)
    np.testing.assert_array_equal(a.scaler.scale_, b.scaler.scale_)
    pd.testing.assert_frame_equal(twin.a["pca_scores"], twin.b["pca_scores"])


# ---- test 17: label rules ----
def test_t17_future_data_cannot_change_label_rules(twin):
    a, b = twin.a["label_rules"], twin.b["label_rules"]
    assert a.score_cuts == b.score_cuts and a.n_training_rows == b.n_training_rows and a.fit_end == b.fit_end
    for name in a.reference:
        np.testing.assert_array_equal(a.reference[name], b.reference[name])
    pd.testing.assert_series_equal(twin.a["labels"], twin.b["labels"])


# ---- test 18: LDA ----
def test_t18_future_data_cannot_change_lda(twin):
    a, b = twin.a["lda_model"], twin.b["lda_model"]
    np.testing.assert_array_equal(a.lda.scalings_, b.lda.scalings_)
    np.testing.assert_array_equal(a.lda.xbar_, b.lda.xbar_)
    np.testing.assert_array_equal(a.lda.means_, b.lda.means_)
    np.testing.assert_array_equal(a.scaler.mean_, b.scaler.mean_)
    pd.testing.assert_frame_equal(twin.a["lda_output"], twin.b["lda_output"])


# ---- test 19: profiles and similarity ----
def test_t19_future_data_cannot_change_profiles_or_similarity(twin):
    pd.testing.assert_frame_equal(twin.a["profiles"], twin.b["profiles"])
    pd.testing.assert_frame_equal(twin.a["similarity_matrix"], twin.b["similarity_matrix"])


# ---- test 20: future returns cannot influence selection ----
def test_t20_a_stock_that_booms_in_the_future_is_not_picked_because_of_it(twin):
    stocks = twin.stocks.copy()
    for boom in ("TCS", "RELIANCE", "SBIN"):  # one stock at a time: a huge future rally or crash
        for factor in (10.0, 0.1):
            altered = stocks.copy()
            hit = (altered["symbol"] == boom) & (altered["date"] >= CUT)
            for column in ("open", "high", "low", "close"):
                altered.loc[hit, column] = altered.loc[hit, column] * factor
            changed = run_single_rebalance(altered, twin.market, CUT, UNIVERSE, basket_size=6)
            pd.testing.assert_frame_equal(changed["basket_result"].basket, twin.a["basket_result"].basket)


def test_t20_the_basket_generator_only_ever_receives_training_period_information(dev, monkeypatch):
    stocks, market = dev
    seen = {}
    original = bt.generate_basket

    def spy(lda_data, similarity_matrix, universe, *args, **kwargs):
        seen["lda_dates"] = pd.to_datetime(lda_data["date"])
        seen["columns"] = list(lda_data.columns)
        seen["similarity_columns"] = list(similarity_matrix.columns)
        return original(lda_data, similarity_matrix, universe, *args, **kwargs)

    monkeypatch.setattr(bt, "generate_basket", spy)
    run_single_rebalance(stocks, market, CUT, UNIVERSE, basket_size=6)
    assert seen["lda_dates"].max() < pd.Timestamp(CUT)
    assert not any("return" in c.lower() or "close" in c.lower() or "price" in c.lower() for c in seen["columns"])
    assert set(seen["similarity_columns"]) <= set(UNIVERSE["symbol"])


def test_t20_features_are_built_only_from_rows_before_the_rebalance_date(dev, monkeypatch):
    stocks, market = dev
    seen = {}
    original = bt.build_features

    def spy(stock_data, market_data):
        seen["stock_max"], seen["market_max"] = stock_data["date"].max(), market_data["date"].max()
        return original(stock_data, market_data)

    monkeypatch.setattr(bt, "build_features", spy)
    run_single_rebalance(stocks, market, CUT, UNIVERSE, basket_size=6)
    assert seen["stock_max"] < pd.Timestamp(CUT) and seen["market_max"] < pd.Timestamp(CUT)  # the future is never even handed over


# ---- test 21: the scaler is fitted on training data only ----
def test_t21_scaler_and_pca_are_fitted_on_training_rows_only(twin, dev):
    stocks, market = dev
    a = twin.a
    cut = pd.Timestamp(CUT)
    assert a["pca_model"].fit_end < cut and a["lda_model"].fit_end < cut and a["label_rules"].fit_end < cut
    assert a["training_features"]["date"].max() < cut
    assert a["pca_model"].n_training_rows == len(a["training_features"]) == a["lda_model"].n_training_rows
    # the full-history model is different, so the equality below is not a coincidence
    from src.features import build_features, clean_feature_data

    everything = clean_feature_data(build_features(stocks, market))
    full = fit_pca(everything, 5)
    assert not np.allclose(full.scaler.mean_, a["pca_model"].scaler.mean_)
    # and it is exactly the model you get from the training rows alone
    alone = fit_pca(a["training_features"], 5)
    np.testing.assert_array_equal(alone.scaler.mean_, a["pca_model"].scaler.mean_)
    np.testing.assert_array_equal(alone.pca.components_, a["pca_model"].pca.components_)


# ---- test 22: chronology ----
def test_t22_every_rebalance_trains_strictly_before_it_and_tests_strictly_after(short_run):
    done = [r for r in short_run["rebalances"] if r["status"] == "rebalanced"]
    assert len(done) == 8
    for i, record in enumerate(done):
        assert record["last_training_date"] < record["rebalance_date"] <= record["period_start"] <= record["period_end"]
        if i + 1 < len(done):
            assert record["period_end"] < done[i + 1]["rebalance_date"]  # rebalance_date < test_period_end of the next, never overlapping
            assert record["rebalance_date"] < done[i + 1]["rebalance_date"]


def test_t22_test_observations_are_never_in_the_training_data(dev):
    stocks, market = dev
    built = run_single_rebalance(stocks, market, CUT, UNIVERSE, basket_size=6)
    training_dates = set(built["training_features"]["date"])
    test_dates = set(stocks.loc[stocks["date"] >= CUT, "date"])
    assert training_dates.isdisjoint(test_dates)
    assert built["last_training_date"] == stocks.loc[stocks["date"] < CUT, "date"].max()
    assert built["n_training_days"] == stocks.loc[stocks["date"] < CUT, "date"].nunique()


def test_t22_rebalancing_on_a_later_date_uses_more_training_rows(dev):
    stocks, market = dev
    early = run_single_rebalance(stocks, market, "2019-04-01", UNIVERSE, basket_size=6)
    late = run_single_rebalance(stocks, market, "2019-07-01", UNIVERSE, basket_size=6)
    assert late["n_training_rows"] > early["n_training_rows"]
    assert early["pca_model"].fit_end < late["pca_model"].fit_end  # nothing is carried over from an earlier date


# ===========================================================================================
# D. MISSING PRICES, ERRORS, OUTPUT, ARCHITECTURE
# ===========================================================================================
def test_a_held_stock_with_a_missing_price_is_stopped_and_nothing_is_filled():
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {"A": [100, 101, 102, 103, 104, 105.0], "B": [100, 110, np.nan, 130, 140, 150.0]}, index=dates
    )
    out = calculate_portfolio_returns(prices, ["A", "B"], dates[1:])
    # day 2: both valid. day 3: B missing -> stopped. day 4: B's price returns but it stays stopped (no gap return invented)
    assert out["n_active"].tolist() == [2, 1, 1, 1, 1]
    a_returns = prices["A"].pct_change().iloc[1:].to_numpy()
    assert out["daily_return"].iloc[0] == pytest.approx((a_returns[0] + 0.10) / 2)
    assert out["daily_return"].iloc[1:].to_numpy() == pytest.approx(a_returns[1:])
    assert out["stopped"] == [{"symbol": "B", "stop_date": dates[2]}]


def test_a_missing_close_the_day_before_a_period_stops_the_stock_immediately():
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame({"A": [100, 101, 102, 103, 104.0], "B": [100, np.nan, 120, 121, 122.0]}, index=dates)
    out = calculate_portfolio_returns(prices, ["A", "B"], dates[2:])  # first period day 3 needs B's day-2 close
    assert out["n_active"].tolist() == [1, 1, 1] and out["stopped"][0]["stop_date"] == dates[2]


def test_when_every_stock_is_stopped_the_return_is_zero_cash():
    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"A": [100, np.nan, 102, 103.0]}, index=dates)
    out = calculate_portfolio_returns(prices, ["A"], dates[1:])
    assert out["daily_return"].tolist() == [0.0, 0.0, 0.0] and out["n_active"].tolist() == [0, 0, 0]


def test_the_missing_price_policy_is_deterministic():
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"A": [1, 2, np.nan, 4, 5, 6.0], "B": [1, 1.1, 1.2, 1.3, np.nan, 1.5]}, index=dates)
    first = calculate_portfolio_returns(prices, ["A", "B"], dates[1:])
    second = calculate_portfolio_returns(prices, ["A", "B"], dates[1:])
    pd.testing.assert_series_equal(first["daily_return"], second["daily_return"])
    assert first["stopped"] == second["stopped"]


def test_a_stock_without_a_close_on_the_last_training_day_cannot_be_selected(dev):
    stocks, market = dev
    day_before = stocks.loc[stocks["date"] < CUT, "date"].max()
    damaged = stocks[~((stocks["symbol"] == "INFY") & (stocks["date"] == day_before))]
    built = run_single_rebalance(damaged, market, CUT, UNIVERSE, basket_size=6)
    assert "INFY" not in built["tradable_symbols"] and len(built["tradable_symbols"]) == 9
    assert "INFY" not in built["basket_result"].basket["symbol"].tolist()


def test_the_backtest_reports_stocks_stopped_by_missing_prices(dev):
    stocks, market = dev
    gap = stocks[~((stocks["symbol"] == "SBIN") & (stocks["date"] == "2019-02-14"))]
    result = run_backtest(gap, market, UNIVERSE, basket_size=10, end_date="2019-06-30")
    events = [e for e in result["missing_data_events"] if e["symbol"] == "SBIN"]
    assert events and events[0]["stop_date"] == pd.Timestamp("2019-02-14") and events[0]["rebalance_date"] == pd.Timestamp("2019-01-01")
    assert result["data_info"]["days_without_holdings"] == 0


def test_a_period_helper_errors_are_clear():
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="empty"):
        calculate_portfolio_returns(prices, [], prices.index)
    with pytest.raises(ValueError, match="No price history for: Z"):
        calculate_portfolio_returns(prices, ["Z"], prices.index)
    with pytest.raises(ValueError, match="not in the price table"):
        calculate_portfolio_returns(prices, ["A"], pd.DatetimeIndex(["2030-01-01"]))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"capital": 0}, "capital"),
        ({"capital": -5}, "capital"),
        ({"basket_size": 2}, "basket_size"),
        ({"basket_size": 11}, "larger than the number of universe stocks"),
        ({"frequency": "weekly"}, "Unknown rebalance frequency"),
        ({"similarity_threshold": 2}, "similarity_threshold"),
        ({"min_history_days": 30}, "min_history_days"),
        ({"n_components": 0}, "n_components"),
        ({"start_date": "not a date"}, "not a valid date"),
        ({"start_date": "2021-01-01", "end_date": "2020-01-01"}, "after end_date"),
    ],
)
def test_bad_settings_raise_a_clear_error(dev, kwargs, message):
    stocks, market = dev
    base = {"basket_size": 6, "end_date": "2019-03-31"}
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        run_backtest(stocks, market, UNIVERSE, **base)


def test_bad_tables_raise_a_clear_error(dev):
    stocks, market = dev
    with pytest.raises(ValueError, match="stock_data must be a non-empty DataFrame"):
        run_backtest(stocks.iloc[0:0], market, UNIVERSE)
    with pytest.raises(ValueError, match="missing column"):
        run_backtest(stocks.drop(columns="close"), market, UNIVERSE)
    other = pd.DataFrame({"symbol": ["ZZZ"], "cap_category": ["Large Cap"]})
    with pytest.raises(ValueError, match="None of the universe symbols"):
        run_backtest(stocks, market, other)


def test_a_rebalance_that_cannot_be_built_is_explained_when_nothing_else_exists(dev):
    stocks, market = dev
    day_before = stocks.loc[stocks["date"] < "2019-01-01", "date"].max()
    damaged = stocks[~((stocks["symbol"] == "INFY") & (stocks["date"] == day_before))]  # only 9 tradable stocks
    with pytest.raises(ValueError, match="First problem: .*larger than the number of eligible stocks"):
        run_backtest(damaged, market, UNIVERSE, basket_size=10, end_date="2019-03-31")


def test_the_universe_restricts_which_stocks_can_be_held(dev):
    stocks, market = dev
    small = UNIVERSE[UNIVERSE["symbol"].isin(["AXISBANK", "HDFCBANK", "ITC", "LT", "SBIN", "TCS"])]
    result = run_backtest(stocks, market, small, basket_size=3, end_date="2019-06-30")
    assert set(result["rebalance_history"]["symbol"]) <= set(small["symbol"])
    assert result["data_info"]["n_stocks"] == 6


# ---- outputs ----
def test_outputs_have_the_exact_columns_and_keys(short_run, tmp_path):
    paths = save_backtest_outputs(short_run, tmp_path / "r.csv", tmp_path / "h.csv", tmp_path / "s.json")
    results = pd.read_csv(paths["results"])
    history = pd.read_csv(paths["rebalances"])
    assert list(results.columns) == ["date", "portfolio_value", "daily_return", "benchmark_value", "benchmark_return"]
    assert list(history.columns) == ["rebalance_date", "symbol", "cap_category", "behavior_class", "weight", "allocation", "reason"]
    assert len(results) == len(short_run["equity_curve"]) and len(history) == 8 * 6
    summary = json.loads(paths["summary"].read_text())
    assert list(summary) == SUMMARY_KEYS
    assert summary["final_portfolio_value"] == pytest.approx(short_run["summary"]["final_portfolio_value"])
    assert results["date"].iloc[0] == "2019-01-01"


def test_the_summary_json_is_strict_json_even_when_sharpe_is_undefined(short_run, tmp_path):
    broken = dict(short_run)
    broken["summary"] = {**short_run["summary"], "sharpe_ratio": None, "benchmark_sharpe_ratio": float("nan")}
    paths = save_backtest_outputs(broken, tmp_path / "r.csv", tmp_path / "h.csv", tmp_path / "s.json")
    text = paths["summary"].read_text()
    assert "NaN" not in text and json.loads(text)["sharpe_ratio"] is None and json.loads(text)["benchmark_sharpe_ratio"] is None


def test_the_result_converts_to_json_for_a_future_api(short_run):
    payload = result_to_jsonable(short_run)
    text = json.dumps(payload)  # must not raise
    again = json.loads(text)
    assert again["summary"]["initial_capital"] == 100_000
    assert again["equity_curve"][0]["date"] == "2019-01-01" and again["data_info"]["start_date"] == "2018-01-01"
    assert again["rebalances"][0]["rebalance_date"] == "2019-01-01"


def test_charts_are_written(short_run, tmp_path):
    equity = plot_equity_curve(short_run, tmp_path / "plots" / "e.png", note="Synthetic")
    drawdown = plot_drawdown(short_run, tmp_path / "plots" / "d.png", note="Synthetic")
    for path in (equity, drawdown):
        assert path.exists() and path.stat().st_size > 5_000 and path.read_bytes()[:4] == b"\x89PNG"


def test_the_report_has_the_required_sections(short_run):
    text = format_backtest_report(short_run, synthetic=True, benchmark_name="Synthetic market index")
    for expected in ["=== WALK-FORWARD BACKTEST ===", "Initial capital: ₹100,000", "Basket size: 6", "Rebalance frequency: Quarterly",
                     "Trading days: 2088", "Stocks: 10", "=== STRATEGY ===", "=== BENCHMARK ===", "Sharpe ratio:", "Max drawdown:",
                     "synthetic market index is the benchmark", "Synthetic pipeline validation only", "Transaction costs = 0"]:
        assert expected in text
    assert len(text.splitlines()) < 60  # concise: never one line per day


# ---- command line ----
@pytest.fixture
def cli_config(tmp_path):
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    config["data"]["prices_file"] = str(PROJECT_ROOT / config["data"]["prices_file"])
    config["data"]["market_index_file"] = str(PROJECT_ROOT / config["data"]["market_index_file"])
    config["backtest"].update(
        end_date="2019-06-30",
        results_file="out/results.csv", rebalance_file="out/history.csv", summary_file="out/summary.json",
        equity_plot="out/e.png", drawdown_plot="out/d.png",
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_command_line_runs_and_writes_everything(cli_config, capsys):
    main(["--config", str(cli_config), "--capital", "50000", "--basket-size", "6"])
    text = capsys.readouterr().out
    assert "Initial capital: ₹50,000" in text and "Basket size: 6" in text and "=== STRATEGY ===" in text
    out = cli_config.parent / "out"
    for name in ("results.csv", "history.csv", "summary.json", "e.png", "d.png"):
        assert (out / name).exists()
    assert json.loads((out / "summary.json").read_text())["initial_capital"] == 50_000
    assert pd.read_csv(out / "history.csv").groupby("rebalance_date").size().eq(6).all()


@pytest.mark.parametrize("flags", [["--basket-size", "99"], ["--capital", "-1"], ["--frequency", "daily"], ["--start-date", "nope"]])
def test_command_line_rejects_bad_input(cli_config, flags):
    with pytest.raises(SystemExit) as stop:
        main(["--config", str(cli_config), *flags])
    assert stop.value.code == 2


# ---- architecture ----
def _module_tree():
    return ast.parse(Path(bt.__file__).read_text(encoding="utf-8"))


def test_the_backtest_does_not_touch_angel_one_or_credentials():
    tree = _module_tree()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    banned = ("smartapi", "smartapi_python", "angel", "logzero", "pyotp", "dotenv", "requests", "os", "socket")
    assert not [m for m in imported if m.split(".")[0].lower() in banned or "angel" in m.lower()]
    source = Path(bt.__file__).read_text(encoding="utf-8").lower()
    for secret in ("api_key", "client_id", "totp", "environ", "getenv", "password"):
        assert secret not in source


def test_only_the_command_line_function_prints():
    tree = _module_tree()
    printers = set()
    for function in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                printers.add(function.name)
    assert printers == {"main"}


def test_the_core_orchestrates_the_existing_modules_instead_of_duplicating_them():
    source = Path(bt.__file__).read_text(encoding="utf-8")
    for needed in ("fit_pca", "fit_label_rules", "fit_lda", "create_stock_profiles", "calculate_similarity", "generate_basket", "build_features"):
        assert needed in source
    for forbidden in ("StandardScaler", "LinearDiscriminantAnalysis", "cosine_similarity", "from sklearn"):
        assert forbidden not in source
