"""
Phase 7: walk-forward historical backtesting.

THE QUESTION
    If this basket method had been run quarter by quarter in the past, using only what was known at
    the time, how would the baskets have behaved compared with the market index?
    It is a test of the METHOD'S BEHAVIOUR on the data supplied. It is not a forecast, not a claim that
    the method is better than anything, and not investment advice.

HOW A WALK-FORWARD BACKTEST WORKS
    1. Pick rebalance dates (default: the first trading day of every quarter that has enough history).
    2. At each rebalance date R, look ONLY at rows dated before R:
         raw prices -> Phase 2 features -> scaler + PCA -> label rules -> LDA
         -> stock profiles -> similarity matrix -> Phase 6 basket (equal weights).
       Every one of those is fitted on that training slice alone and never reused at the next date.
    3. Hold that basket from R until the next rebalance date and record the daily returns.
    4. Join the holding periods into one equity curve and compare it with the benchmark.

NO LOOK-AHEAD BIAS (the rule that matters most)
    Look-ahead bias means using information that did not exist yet at the decision time. Here the
    decision at R uses rows with date < R only. The slice is cut BEFORE any feature is computed, so even
    a rolling-window mistake could not reach the future. Eight tests prove it by changing the future and
    checking that nothing earlier moves (see tests/test_backtest.py).

    Timing: the basket is chosen from data up to the last training day (the close before R) and earns the
    return of every trading day from R onward. The first return it earns is close(R) / close(R-1) - 1,
    so it is, in effect, bought at the close of R-1, which is exactly when its information became available.

MISSING PRICES (conservative, deterministic, never invents a price)
    * Selection: a stock can only be chosen if it has a real close on the last training day.
    * Holding: if a held stock has no usable return on some day (its close or the previous close is
      missing), it is STOPPED for the rest of that holding period and the remaining stocks stay equally
      weighted. Nothing is forward-filled or back-filled. If every stock were stopped the day's
      return is 0 (cash) and the day is counted in `missing_data_events`/`days_without_holdings`.

SIMPLIFICATIONS (stated, not hidden)
    * Transaction costs = 0 and slippage = 0.  No cost model.
    * Daily basket return = mean of the held stocks' daily returns (equal weights each day, as specified).
    * Risk-free rate = 0 in the Sharpe ratio.
    * The cap classes and the universe are fixed (no historical index membership), so survivorship bias
      is possible on real data.
    * The development data is SYNTHETIC. Results on it only validate that the pipeline runs correctly.

FOR THE FUTURE API
    Every function returns plain Python objects (dicts, DataFrames, lists). Only `main` prints.
    `result_to_jsonable` turns a result into JSON-ready data, so a FastAPI endpoint can later call
    `run_backtest` without a rewrite. This module knows nothing about Angel One: it consumes the standard
    price tables from src/data_loader.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from src.basket import (
    DEFAULT_BASKET_SIZE,
    DEFAULT_CAPITAL,
    DEFAULT_SIMILARITY_THRESHOLD,
    _check_basket_size,
    _check_capital,
    _check_threshold,
    format_rupees,
    generate_basket,
)
from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import MIN_HISTORY_ROWS, build_features, clean_feature_data
from src.labels import create_behavior_labels, fit_label_rules
from src.lda_model import fit_lda, transform_lda
from src.pca_model import _BLUE, _GRID, _INK, _INK_2, _MUTED, _SURFACE, _style_axes, _title, fit_pca, transform_pca
from src.similarity import calculate_similarity, create_stock_profiles
from src.universe import get_development_universe

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.25
DEFAULT_MIN_HISTORY_DAYS = 252
DEFAULT_FREQUENCY = "quarterly"
DEFAULT_N_COMPONENTS = 5
REBALANCE_FREQUENCIES = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}  # months per holding period

RESULT_COLUMNS = ["date", "portfolio_value", "daily_return", "benchmark_value", "benchmark_return"]
HISTORY_COLUMNS = ["rebalance_date", "symbol", "cap_category", "behavior_class", "weight", "allocation", "reason"]
SUMMARY_KEYS = [
    "initial_capital",
    "final_portfolio_value",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "benchmark_cumulative_return",
    "benchmark_annualized_return",
    "benchmark_volatility",
    "benchmark_sharpe_ratio",
    "benchmark_max_drawdown",
]

SYNTHETIC_NOTICE = (
    "Synthetic pipeline validation only. The data is synthetic and the benchmark is a synthetic market index, "
    "so these numbers say nothing about real markets."
)
GENERAL_NOTICE = (
    "Historical simulation on the supplied data only. It does not predict future returns and is not investment advice."
)
ASSUMPTIONS = (
    "Transaction costs = 0, slippage = 0, risk-free rate = 0. No guarantee of any outcome."
)


# ---------------------------------------------------------------------------
# Input checks and small helpers
# ---------------------------------------------------------------------------
def _check_frequency(frequency) -> str:
    if not isinstance(frequency, str) or frequency.lower() not in REBALANCE_FREQUENCIES:
        raise ValueError(f"Unknown rebalance frequency {frequency!r}. Choose one of: {', '.join(REBALANCE_FREQUENCIES)}.")
    return frequency.lower()


def _check_min_history(min_history_days) -> int:
    if isinstance(min_history_days, bool) or not isinstance(min_history_days, (int, np.integer)):
        raise ValueError("min_history_days must be a whole number of trading days.")
    if min_history_days <= MIN_HISTORY_ROWS:
        raise ValueError(
            f"min_history_days ({min_history_days}) must be larger than the {MIN_HISTORY_ROWS}-day feature warm-up, "
            "otherwise there would be no usable training rows."
        )
    return int(min_history_days)


def _check_components(n_components) -> int:
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError("n_components must be a whole number of at least 1.")
    return int(n_components)


def _to_timestamp(value, name: str):
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} {value!r} is not a valid date (use YYYY-MM-DD).") from error
    if pd.isna(stamp):
        raise ValueError(f"{name} {value!r} is not a valid date (use YYYY-MM-DD).")
    return stamp


def _check_price_table(table, name: str) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame) or len(table) == 0:
        raise ValueError(f"{name} must be a non-empty DataFrame.")
    missing = [c for c in ("date", "symbol", "close") if c not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing column(s): {', '.join(missing)}.")
    clean = table.copy()
    clean["date"] = pd.to_datetime(clean["date"])
    return clean.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def _close_wide(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Close prices as a date x symbol table. A symbol with no row on a date shows up as NaN (never filled)."""
    return stock_data.pivot(index="date", columns="symbol", values="close").sort_index()


def _period_key(date: pd.Timestamp, months: int) -> tuple:
    return (date.year, (date.month - 1) // months)


# ---------------------------------------------------------------------------
# Rebalance dates
# ---------------------------------------------------------------------------
def generate_rebalance_dates(
    trading_dates,
    frequency: str = DEFAULT_FREQUENCY,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    start_date=None,
    end_date=None,
) -> list:
    """
    Rebalance on the FIRST TRADING DAY of every period (quarter by default) that actually appears in `trading_dates`.

    A date only counts if at least `min_history_days` trading days exist strictly before it, so the first
    rebalance waits until the 60-day features and the models have enough training rows. When the first
    trading day of a period is too early, that period is skipped (we never rebalance on a later day of the
    period). `start_date` / `end_date` optionally restrict the dates that may be used.
    """
    months = REBALANCE_FREQUENCIES[_check_frequency(frequency)]
    min_history_days = _check_min_history(min_history_days)
    start, end = _to_timestamp(start_date, "start_date"), _to_timestamp(end_date, "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError(f"start_date ({start.date()}) is after end_date ({end.date()}).")
    dates = pd.DatetimeIndex(pd.to_datetime(list(trading_dates))).unique().sort_values()

    chosen, seen = [], set()
    for position, date in enumerate(dates):
        key = _period_key(date, months)
        if key in seen:
            continue
        seen.add(key)  # the first trading day of the period is the only candidate
        if position < min_history_days:
            continue
        if start is not None and date < start:
            continue
        if end is not None and date > end:
            continue
        chosen.append(date)
    return chosen


# ---------------------------------------------------------------------------
# One rebalance: everything is learned from rows BEFORE the rebalance date
# ---------------------------------------------------------------------------
def run_single_rebalance(
    stock_data: pd.DataFrame,
    market_data: pd.DataFrame,
    rebalance_date,
    universe: pd.DataFrame,
    basket_size: int = DEFAULT_BASKET_SIZE,
    capital: float = DEFAULT_CAPITAL,
    n_components: int = DEFAULT_N_COMPONENTS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> dict:
    """
    Build the basket for one rebalance date using only rows dated before it.

    `stock_data` and `market_data` may contain the whole history (even the future): they are cut at
    `rebalance_date` first, before any feature is computed. Returns a dict holding the basket and every
    fitted object (so tests and later API code can inspect them). Raises ValueError when the history is
    too short or the training slice cannot support the models.
    """
    stock_data = _check_price_table(stock_data, "stock_data")
    market_data = _check_price_table(market_data, "market_data")
    rebalance_date = _to_timestamp(rebalance_date, "rebalance_date")
    min_history_days = _check_min_history(min_history_days)
    n_components = _check_components(n_components)

    in_universe = set(universe["symbol"].astype(str))
    train_stocks = stock_data[(stock_data["date"] < rebalance_date) & stock_data["symbol"].astype(str).isin(in_universe)]
    train_market = market_data[market_data["date"] < rebalance_date]
    training_dates = train_stocks["date"].drop_duplicates().sort_values()
    if len(training_dates) < min_history_days:
        raise ValueError(
            f"Not enough history before {rebalance_date.date()}: {len(training_dates)} trading days, "
            f"at least {min_history_days} are needed."
        )
    last_training_date = training_dates.iloc[-1]

    features = clean_feature_data(build_features(train_stocks, train_market))
    if len(features) == 0:
        raise ValueError(f"No usable feature rows before {rebalance_date.date()}.")

    pca_model = fit_pca(features, n_components)
    pca_scores = transform_pca(features, pca_model)
    rules = fit_label_rules(features)
    labels = create_behavior_labels(features, rules)
    lda_model = fit_lda(features, labels)
    lda_output = transform_lda(features, lda_model, labels)
    profiles = create_stock_profiles(pca_scores)
    similarity_matrix = calculate_similarity(profiles)

    # Only stocks with a real close on the last training day can be bought (training information only).
    last_close = _close_wide(train_stocks).loc[last_training_date]
    tradable = sorted(str(s) for s in last_close.index[last_close.notna()])
    lda_tradable = lda_output[lda_output["symbol"].astype(str).isin(tradable)]
    similarity_tradable = similarity_matrix.loc[
        [s for s in similarity_matrix.index if str(s) in tradable], [s for s in similarity_matrix.columns if str(s) in tradable]
    ]
    basket_result = generate_basket(
        lda_tradable, similarity_tradable, universe, basket_size, capital, similarity_threshold
    )
    return {
        "rebalance_date": rebalance_date,
        "last_training_date": last_training_date,
        "n_training_days": int(len(training_dates)),
        "n_training_rows": int(len(features)),
        "training_features": features,
        "pca_model": pca_model,
        "pca_scores": pca_scores,
        "label_rules": rules,
        "labels": labels,
        "lda_model": lda_model,
        "lda_output": lda_output,
        "profiles": profiles,
        "similarity_matrix": similarity_matrix,
        "tradable_symbols": tradable,
        "basket_result": basket_result,
    }


# ---------------------------------------------------------------------------
# Returns, benchmark, drawdown, metrics
# ---------------------------------------------------------------------------
def calculate_portfolio_returns(prices: pd.DataFrame, symbols, period_dates) -> dict:
    """
    Daily equal-weight return of the held `symbols` over `period_dates`.

    A stock's return on day d is close(d) / close(previous trading day) - 1. The previous day comes from the
    full `prices` table, so the first day of a period uses the last day of the training window.
    Missing-price policy: once a stock has no usable return, it is STOPPED for the rest of the period (never
    filled); the day's return is the mean over the stocks still active. If none are active the return is 0.

    Returns {"daily_return": Series, "n_active": Series, "stopped": list of {symbol, stop_date}}.
    """
    symbols = [str(s) for s in symbols]
    if not symbols:
        raise ValueError("symbols is empty: there is nothing to hold.")
    unknown = [s for s in symbols if s not in {str(c) for c in prices.columns}]
    if unknown:
        raise ValueError(f"No price history for: {', '.join(unknown)}.")
    prices = prices.sort_index()
    period_dates = pd.DatetimeIndex(period_dates)
    if len(period_dates) == 0:
        raise ValueError("period_dates is empty.")
    if not period_dates.isin(prices.index).all():
        raise ValueError("period_dates contains dates that are not in the price table.")

    held = prices[symbols]
    returns = (held / held.shift(1) - 1.0).loc[period_dates]
    valid = returns.notna()
    active = valid.astype(int).cummin() == 1  # once a stock is stopped it stays stopped
    daily = returns.where(active).mean(axis=1).fillna(0.0)
    n_active = active.sum(axis=1).astype(int)
    stopped = []
    for symbol in symbols:
        if not bool(active[symbol].iloc[-1]):
            stop_date = active.index[~active[symbol].to_numpy()][0]
            stopped.append({"symbol": symbol, "stop_date": stop_date})
    return {"daily_return": daily.rename("daily_return"), "n_active": n_active.rename("n_active"), "stopped": stopped}


def calculate_benchmark(market_close: pd.Series, dates, initial_capital: float = DEFAULT_CAPITAL) -> pd.DataFrame:
    """
    Benchmark daily returns and value on `dates`: the market index, same dates, same starting capital.
    A return needs the close of the day and of the previous trading day; a missing one is an error (never invented).
    """
    initial_capital = _check_capital(initial_capital)
    close = market_close.sort_index()
    dates = pd.DatetimeIndex(dates)
    if not dates.isin(close.index).all():
        raise ValueError("The market index has no close for some backtest dates.")
    returns = (close / close.shift(1) - 1.0).loc[dates]
    if returns.isna().any():
        raise ValueError("The market index has a missing price inside the backtest, so its return cannot be computed.")
    value = initial_capital * (1.0 + returns).cumprod()
    return pd.DataFrame({"benchmark_return": returns.to_numpy(), "benchmark_value": value.to_numpy()}, index=dates)


def calculate_drawdown(values) -> pd.Series:
    """Drawdown = value / running peak - 1 (0 at a new high, negative below it)."""
    values = pd.Series(values, dtype="float64")
    if len(values) == 0:
        raise ValueError("values is empty.")
    return values / values.cummax() - 1.0


def calculate_metrics(daily_returns, years: float) -> dict:
    """
    The five headline metrics from a series of daily returns.

    cumulative_return   = final / initial - 1                       (compounded daily returns)
    annualized_return   = (1 + cumulative) ** (1 / years) - 1       (years = real calendar duration)
    annualized_volatility = std(daily returns) * sqrt(252)
    sharpe_ratio        = annualized_return / annualized_volatility (risk-free rate = 0; None if volatility is 0)
    max_drawdown        = minimum of value / running peak - 1       (the start value counts as the first peak)
    """
    returns = pd.Series(daily_returns, dtype="float64")
    if len(returns) < 2:
        raise ValueError("At least 2 daily returns are needed to compute the metrics.")
    if returns.isna().any():
        raise ValueError("daily_returns contains missing values.")
    if not (isinstance(years, (int, float, np.floating)) and years > 0 and math.isfinite(years)):
        raise ValueError("years must be a positive number.")
    growth = (1.0 + returns).cumprod()
    cumulative = float(growth.iloc[-1] - 1.0)
    annualized = float((1.0 + cumulative) ** (1.0 / years) - 1.0)
    volatility = float(returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = annualized / volatility if volatility > 0 else None
    path = pd.concat([pd.Series([1.0]), growth.reset_index(drop=True)], ignore_index=True)
    max_drawdown = float(calculate_drawdown(path).min())
    return {
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_ratio": None if sharpe is None else float(sharpe),
        "max_drawdown": max_drawdown,
    }


# ---------------------------------------------------------------------------
# The full backtest
# ---------------------------------------------------------------------------
def _period_end_dates(all_dates: pd.DatetimeIndex, rebalance_dates: list, end: pd.Timestamp | None):
    """For each rebalance date: the trading days it is held (from the date up to, not including, the next one)."""
    last_allowed = all_dates[-1] if end is None else min(all_dates[-1], end)
    periods = []
    for i, start in enumerate(rebalance_dates):
        stop = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        mask = (all_dates >= start) & (all_dates <= last_allowed)
        if stop is not None:
            mask &= all_dates < stop
        periods.append(all_dates[mask])
    return periods


def run_backtest(
    stock_data: pd.DataFrame,
    market_data: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    capital: float = DEFAULT_CAPITAL,
    basket_size: int = DEFAULT_BASKET_SIZE,
    frequency: str = DEFAULT_FREQUENCY,
    n_components: int = DEFAULT_N_COMPONENTS,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
    start_date=None,
    end_date=None,
) -> dict:
    """
    Walk-forward backtest. Returns a dict (nothing is printed or written):

      summary            the 12 headline numbers (SUMMARY_KEYS)
      equity_curve       DataFrame: date, portfolio_value, daily_return, benchmark_value, benchmark_return
      drawdown           DataFrame: date, portfolio_drawdown, benchmark_drawdown
      rebalance_history  DataFrame: rebalance_date, symbol, cap_category, behavior_class, weight, allocation, reason
      rebalances         one dict per rebalance date (status, holding period, training size, holdings)
      skipped_rebalances dates that could not be built, with the reason
      missing_data_events stocks stopped because of missing prices
      data_info, settings, notices
    """
    capital = _check_capital(capital)
    basket_size = _check_basket_size(basket_size)
    similarity_threshold = _check_threshold(similarity_threshold)
    n_components = _check_components(n_components)
    frequency = _check_frequency(frequency)
    min_history_days = _check_min_history(min_history_days)
    start, end = _to_timestamp(start_date, "start_date"), _to_timestamp(end_date, "end_date")
    stock_data = _check_price_table(stock_data, "stock_data")
    market_data = _check_price_table(market_data, "market_data")
    universe = get_development_universe() if universe is None else universe

    in_universe = set(universe["symbol"].astype(str))
    stock_data = stock_data[stock_data["symbol"].astype(str).isin(in_universe)].reset_index(drop=True)
    if len(stock_data) == 0:
        raise ValueError("None of the universe symbols appear in stock_data.")
    n_available = stock_data["symbol"].nunique()
    if basket_size > n_available:
        raise ValueError(f"basket_size ({basket_size}) is larger than the number of universe stocks that have data ({n_available}).")
    prices = _close_wide(stock_data)
    market_close = market_data.drop_duplicates("date").set_index("date")["close"].sort_index()
    all_dates = pd.DatetimeIndex(prices.index)

    months = REBALANCE_FREQUENCIES[frequency]
    possible_periods = len({_period_key(d, months) for d in all_dates})
    rebalance_dates = generate_rebalance_dates(all_dates, frequency, min_history_days, start, end)
    data_info = {
        "start_date": all_dates[0],
        "end_date": all_dates[-1],
        "trading_days": int(len(all_dates)),
        "n_stocks": int(prices.shape[1]),
        "possible_periods": int(possible_periods),
        "n_rebalance_dates": int(len(rebalance_dates)),
    }
    if not rebalance_dates:
        raise ValueError(
            f"No rebalance is possible: the data has {len(all_dates)} trading days and the first {frequency} rebalance needs "
            f"{min_history_days} days of history before it. Use more data or a smaller min_history_days."
        )

    periods = _period_end_dates(all_dates, rebalance_dates, end)
    held_value = capital
    current_symbols: list | None = None
    return_pieces, history_pieces, rebalances, skipped, stopped_events = [], [], [], [], []
    days_without_holdings = 0

    for rebalance_date, period_dates in zip(rebalance_dates, periods):
        if len(period_dates) == 0:
            continue
        record = {"rebalance_date": rebalance_date, "period_start": period_dates[0], "period_end": period_dates[-1]}
        try:
            built = run_single_rebalance(
                stock_data, market_data, rebalance_date, universe, basket_size, held_value, n_components,
                similarity_threshold, min_history_days,
            )
        except ValueError as error:
            skipped.append({"rebalance_date": rebalance_date, "reason": str(error)})
            record.update(status="skipped_kept_previous_basket" if current_symbols else "skipped_no_basket", reason=str(error),
                          n_holdings=len(current_symbols or []), symbols=list(current_symbols or []))
            rebalances.append(record)
            if current_symbols is None:
                continue  # nothing held yet: these days are not part of the backtest
        else:
            basket = built["basket_result"].basket
            current_symbols = basket["symbol"].tolist()
            history_pieces.append(basket.assign(rebalance_date=rebalance_date)[HISTORY_COLUMNS])
            record.update(status="rebalanced", last_training_date=built["last_training_date"],
                          n_training_rows=built["n_training_rows"], n_holdings=len(current_symbols), symbols=list(current_symbols),
                          capital=float(held_value), notes=list(built["basket_result"].notes))
            rebalances.append(record)

        outcome = calculate_portfolio_returns(prices, current_symbols, period_dates)
        for event in outcome["stopped"]:
            stopped_events.append({"rebalance_date": rebalance_date, **event})
        days_without_holdings += int((outcome["n_active"] == 0).sum())
        return_pieces.append(outcome["daily_return"])
        held_value = held_value * float((1.0 + outcome["daily_return"]).prod())

    if not return_pieces:
        why = f" First problem: {skipped[0]['reason']}" if skipped else ""
        raise ValueError(f"No rebalance could be built from the data, so there is nothing to backtest.{why}")

    daily = pd.concat(return_pieces)
    if len(daily) < 2:
        raise ValueError("The backtest has fewer than 2 trading days; use a longer window.")
    benchmark = calculate_benchmark(market_close, daily.index, capital)
    value = capital * (1.0 + daily).cumprod()
    equity = pd.DataFrame(
        {
            "date": daily.index,
            "portfolio_value": value.to_numpy(),
            "daily_return": daily.to_numpy(),
            "benchmark_value": benchmark["benchmark_value"].to_numpy(),
            "benchmark_return": benchmark["benchmark_return"].to_numpy(),
        }
    )[RESULT_COLUMNS].reset_index(drop=True)

    first_position = all_dates.get_loc(daily.index[0])
    baseline = all_dates[first_position - 1]  # the close before the first return: the moment the money is put to work
    years = (daily.index[-1] - baseline).days / DAYS_PER_YEAR
    strategy = calculate_metrics(daily, years)
    bench = calculate_metrics(benchmark["benchmark_return"], years)
    summary = {
        "initial_capital": float(capital),
        "final_portfolio_value": float(equity["portfolio_value"].iloc[-1]),
        "cumulative_return": strategy["cumulative_return"],
        "annualized_return": strategy["annualized_return"],
        "annualized_volatility": strategy["annualized_volatility"],
        "sharpe_ratio": strategy["sharpe_ratio"],
        "max_drawdown": strategy["max_drawdown"],
        "benchmark_cumulative_return": bench["cumulative_return"],
        "benchmark_annualized_return": bench["annualized_return"],
        "benchmark_volatility": bench["annualized_volatility"],
        "benchmark_sharpe_ratio": bench["sharpe_ratio"],
        "benchmark_max_drawdown": bench["max_drawdown"],
    }
    drawdown = pd.DataFrame(
        {
            "date": equity["date"],
            "portfolio_drawdown": calculate_drawdown(pd.concat([pd.Series([capital]), equity["portfolio_value"]], ignore_index=True)).iloc[1:].to_numpy(),
            "benchmark_drawdown": calculate_drawdown(pd.concat([pd.Series([capital]), equity["benchmark_value"]], ignore_index=True)).iloc[1:].to_numpy(),
        }
    )
    history = pd.concat(history_pieces, ignore_index=True)[HISTORY_COLUMNS] if history_pieces else pd.DataFrame(columns=HISTORY_COLUMNS)
    data_info.update(
        first_rebalance=rebalance_dates[0],
        last_rebalance=rebalance_dates[-1],
        backtest_start=daily.index[0],
        backtest_end=daily.index[-1],
        baseline_date=baseline,
        years=float(years),
        days_without_holdings=days_without_holdings,
    )
    return {
        "summary": summary,
        "equity_curve": equity,
        "drawdown": drawdown,
        "rebalance_history": history,
        "rebalances": rebalances,
        "skipped_rebalances": skipped,
        "missing_data_events": stopped_events,
        "data_info": data_info,
        "settings": {
            "capital": float(capital), "basket_size": int(basket_size), "frequency": frequency,
            "n_components": n_components, "similarity_threshold": float(similarity_threshold),
            "min_history_days": min_history_days, "transaction_cost": 0.0, "slippage": 0.0, "risk_free_rate": 0.0,
        },
        "notices": {"assumptions": ASSUMPTIONS, "benchmark": "Benchmark = the market index supplied with the data."},
    }


# ---------------------------------------------------------------------------
# JSON-ready output (for the future API) and files
# ---------------------------------------------------------------------------
def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def result_to_jsonable(result: dict) -> dict:
    """Turn a run_backtest result into plain JSON data (dates as YYYY-MM-DD, NaN as null)."""
    return _jsonable(result)


def save_backtest_outputs(result: dict, results_file, rebalance_file, summary_file) -> dict:
    """Write the equity-curve CSV, the rebalance-history CSV and the summary JSON. Returns the three paths."""
    paths = {"results": Path(results_file), "rebalances": Path(rebalance_file), "summary": Path(summary_file)}
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    result["equity_curve"].to_csv(paths["results"], index=False, date_format="%Y-%m-%d")
    result["rebalance_history"].to_csv(paths["rebalances"], index=False, date_format="%Y-%m-%d")
    summary = {key: result["summary"][key] for key in SUMMARY_KEYS}
    paths["summary"].write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    return paths


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _save_figure(fig: Figure, path, note: str | None) -> Path:
    if note:
        fig.text(0.01, 0.005, note, color=_MUTED, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    return path


def plot_equity_curve(result: dict, path, note: str | None = None, benchmark_name: str = "Market index") -> Path:
    """Portfolio value and benchmark value over time (both start from the same capital)."""
    equity = result["equity_curve"]
    start = result["data_info"]["baseline_date"]
    capital = result["summary"]["initial_capital"]
    dates = pd.concat([pd.Series([start]), equity["date"]], ignore_index=True)
    portfolio = pd.concat([pd.Series([capital]), equity["portfolio_value"]], ignore_index=True)
    benchmark = pd.concat([pd.Series([capital]), equity["benchmark_value"]], ignore_index=True)

    fig = Figure(figsize=(10, 5.2), facecolor=_SURFACE)
    ax = fig.subplots()
    _style_axes(ax, grid_axis="y")
    ax.plot(dates, benchmark, color=_MUTED, linewidth=1.8, label=benchmark_name)
    ax.plot(dates, portfolio, color=_BLUE, linewidth=2.0, label="Basket strategy")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: format_rupees(value)))
    for series, colour, name in ((portfolio, _BLUE, "Basket"), (benchmark, _INK_2, "Index")):
        ax.annotate(f"{name} {format_rupees(series.iloc[-1])}", (dates.iloc[-1], series.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=colour)
    ax.margins(x=0.02)
    fig.subplots_adjust(right=0.82)
    _title(ax, "Portfolio value vs benchmark",
           f"Walk-forward, {result['settings']['frequency']} rebalancing, equal weights, zero costs. Not investment advice")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=_INK_2)
    return _save_figure(fig, path, note)


def plot_drawdown(result: dict, path, note: str | None = None, benchmark_name: str = "Market index") -> Path:
    """How far the portfolio sat below its previous peak, over time (benchmark as a thin line for context)."""
    table = result["drawdown"]
    fig = Figure(figsize=(10, 4.2), facecolor=_SURFACE)
    ax = fig.subplots()
    _style_axes(ax, grid_axis="y")
    ax.fill_between(table["date"], table["portfolio_drawdown"], 0, color=_BLUE, alpha=0.25, linewidth=0)
    ax.plot(table["date"], table["portfolio_drawdown"], color=_BLUE, linewidth=1.6, label="Basket strategy")
    ax.plot(table["date"], table["benchmark_drawdown"], color=_MUTED, linewidth=1.2, label=benchmark_name)
    ax.axhline(0, color=_GRID, linewidth=0.8)
    ax.set_ylim(top=0.012)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.margins(x=0.02)
    worst = result["summary"]["max_drawdown"]
    _title(ax, "Portfolio drawdown", f"Value vs its running peak. Worst: {worst:.1%}. Not investment advice")
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=_INK_2)
    return _save_figure(fig, path, note)


# ---------------------------------------------------------------------------
# Report text (built here, printed only by main)
# ---------------------------------------------------------------------------
def _pct(value) -> str:
    return "n/a" if value is None else f"{value:+.2%}"


def _date_rows(dates: list, per_row: int = 7, limit: int = 12) -> list:
    text = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    if len(text) > limit:
        text = text[:4] + ["..."] + text[-3:]
    return ["  " + "  ".join(text[i:i + per_row]) for i in range(0, len(text), per_row)]


def format_backtest_report(result: dict, synthetic: bool = False, benchmark_name: str = "Market index") -> str:
    s, info, cfg = result["summary"], result["data_info"], result["settings"]
    equity = result["equity_curve"]
    done = [r for r in result["rebalances"] if r["status"] == "rebalanced"]
    lines = [
        "=== WALK-FORWARD BACKTEST ===",
        f"Initial capital: {format_rupees(s['initial_capital'])}",
        f"Basket size: {cfg['basket_size']}",
        f"Rebalance frequency: {cfg['frequency'].capitalize()}",
        "",
        "Available data:",
        f"  Start: {info['start_date'].date()}",
        f"  End: {info['end_date'].date()}",
        f"  Trading days: {info['trading_days']}",
        f"  Stocks: {info['n_stocks']}",
        f"  Possible {cfg['frequency']} periods in the data: {info['possible_periods']}",
        f"  History needed before the first rebalance: {cfg['min_history_days']} trading days",
        "",
        "Rebalances:",
        f"  Count: {len(done)} (first {info['first_rebalance'].date()}, last {info['last_rebalance'].date()})",
        *_date_rows([r["rebalance_date"] for r in done]),
        f"  Skipped (could not be built): {len(result['skipped_rebalances'])}",
        f"  Stocks stopped by missing prices: {len(result['missing_data_events'])}",
        f"  Backtest period: {info['backtest_start'].date()} to {info['backtest_end'].date()} ({info['years']:.2f} years)",
        "",
        "=== STRATEGY ===",
        f"Final value: {format_rupees(s['final_portfolio_value'])}",
        f"Cumulative return: {_pct(s['cumulative_return'])}",
        f"Annualised return: {_pct(s['annualized_return'])}",
        f"Annualised volatility: {s['annualized_volatility']:.2%}",
        f"Sharpe ratio: {'n/a' if s['sharpe_ratio'] is None else f'{s['sharpe_ratio']:.2f}'}",
        f"Max drawdown: {s['max_drawdown']:.2%}",
        "",
        "=== BENCHMARK ===",
        f"({benchmark_name}; {'synthetic market index is the benchmark' if synthetic else 'index supplied with the data'})",
        f"Final value: {format_rupees(float(equity['benchmark_value'].iloc[-1]))}",
        f"Cumulative return: {_pct(s['benchmark_cumulative_return'])}",
        f"Annualised return: {_pct(s['benchmark_annualized_return'])}",
        f"Annualised volatility: {s['benchmark_volatility']:.2%}",
        f"Sharpe ratio: {'n/a' if s['benchmark_sharpe_ratio'] is None else f'{s['benchmark_sharpe_ratio']:.2f}'}",
        f"Max drawdown: {s['benchmark_max_drawdown']:.2%}",
        "",
        ASSUMPTIONS,
        SYNTHETIC_NOTICE if synthetic else GENERAL_NOTICE,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the multi-cap basket method.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--capital", type=float, default=None, help="starting rupees (default from config, 100000)")
    parser.add_argument("--basket-size", type=int, default=None, help="stocks per basket (default from config, 10)")
    parser.add_argument("--frequency", default=None, help=f"rebalance frequency: {', '.join(REBALANCE_FREQUENCIES)} (default quarterly)")
    parser.add_argument("--similarity-threshold", type=float, default=None, help="redundancy limit, default 0.90")
    parser.add_argument("--start-date", default=None, help="first allowed rebalance date, YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="last day of the test, YYYY-MM-DD")
    args = parser.parse_args(argv)
    try:  # a terminal that cannot show the rupee sign should print '?' rather than crash
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    config = load_config(config_path)
    cfg = config.get("backtest", {})

    def pick(flag, key, default):
        return flag if flag is not None else cfg.get(key, default)

    capital = pick(args.capital, "capital", DEFAULT_CAPITAL)
    basket_size = pick(args.basket_size, "basket_size", DEFAULT_BASKET_SIZE)
    frequency = pick(args.frequency, "rebalance_frequency", DEFAULT_FREQUENCY)
    threshold = pick(args.similarity_threshold, "similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    start_date = pick(args.start_date, "start_date", None)
    end_date = pick(args.end_date, "end_date", None)

    stock_data = get_provider(config, base_dir).get_historical_data()
    market_data = LocalDataProvider(base_dir / config["data"]["market_index_file"]).get_historical_data()
    try:
        result = run_backtest(
            stock_data, market_data, get_development_universe(), capital, basket_size, frequency,
            cfg.get("n_components", DEFAULT_N_COMPONENTS), threshold, cfg.get("min_history_days", DEFAULT_MIN_HISTORY_DAYS),
            start_date, end_date,
        )
    except ValueError as error:
        parser.error(str(error))

    source = str(config.get("data", {}).get("prices_file", ""))
    synthetic = "synthetic" in source.lower()
    print(format_backtest_report(result, synthetic=synthetic, benchmark_name="Synthetic market index" if synthetic else "Market index"))

    paths = save_backtest_outputs(
        result,
        base_dir / cfg.get("results_file", "data/processed/backtest_results.csv"),
        base_dir / cfg.get("rebalance_file", "data/processed/rebalance_history.csv"),
        base_dir / cfg.get("summary_file", "data/processed/backtest_summary.json"),
    )
    note = "Synthetic data and a synthetic market index. Pipeline validation only." if synthetic else None
    name = "Synthetic market index" if synthetic else "Market index"
    equity_plot = plot_equity_curve(result, base_dir / cfg.get("equity_plot", "data/processed/plots/backtest_equity_curve.png"), note, name)
    drawdown_plot = plot_drawdown(result, base_dir / cfg.get("drawdown_plot", "data/processed/plots/backtest_drawdown.png"), note, name)
    print()
    print(f"Results saved to: {paths['results']}")
    print(f"Rebalance history saved to: {paths['rebalances']}")
    print(f"Summary saved to: {paths['summary']}")
    print(f"Charts saved to: {equity_plot} and {drawdown_plot}")


if __name__ == "__main__":
    main()
