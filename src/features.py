"""
Phase 2: feature engineering.

What this file does, in plain words
-----------------------------------
Takes daily prices (the Phase 1 table) and turns each stock-day into a row of
13 numbers that DESCRIBE how the stock has behaved recently: how much it moved,
how bumpy it was, how it sits against its averages, how it moves with the market,
and how heavily it trades.

These numbers describe the PAST. They are not forecasts, and nothing here claims
they predict future returns. Later phases use them like this:
  PCA        -> finds the main patterns shared by these 13 columns
  LDA        -> separates predefined behaviour classes
  Similarity -> compares stocks in that learned space
These stages stay separate; this file does none of them.

THE ONE RULE (no leakage)
-------------------------
The feature on date t may only use data from date t or EARLIER.
Every calculation below is a "trailing" window (it looks backwards), and nothing
is ever shifted backwards in time, forward-filled or back-filled.
tests/test_features.py proves it: it changes future prices and checks that earlier
features do not move, and it checks that cutting the data off at date t gives the
same feature values at t as running on the full history.

Units: returns and drawdown are decimals (0.05 means +5%, -0.12 means -12%).
"20 days" always means 20 trading rows of data, not 20 calendar days.

Run it:   python -m src.features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import LocalDataProvider, get_provider, load_config

# ---------------------------------------------------------------------------
# Settings (change here, nowhere else)
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252  # used only to annualise volatility
DRAWDOWN_WINDOW = 60  # max_drawdown looks at the last 60 trading days
RSI_PERIOD = 14

# Column names are a contract: Phase 3 (PCA) and Phase 4 (LDA) rely on them.
FEATURE_COLUMNS = [
    "return_5d",
    "return_20d",
    "return_60d",
    "volatility_20d",
    "volatility_60d",
    "price_ma20_ratio",
    "price_ma50_ratio",
    "rsi_14",
    "max_drawdown",
    "beta_60d",
    "market_correlation_60d",
    "avg_volume_20d",
    "volume_change",
]
ID_COLUMNS = ["date", "symbol", "close"]
OUTPUT_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS

# How many rows of history each feature needs before its first valid value.
# Used for documentation and checked in the tests.
# (A return needs the price N days ago, so return_60d needs 61 prices -> first valid on row index 60.)
FIRST_VALID_ROW = {
    "return_5d": 5,
    "return_20d": 20,
    "return_60d": 60,
    "volatility_20d": 20,
    "volatility_60d": 60,
    "price_ma20_ratio": 19,
    "price_ma50_ratio": 49,
    "rsi_14": 14,
    "max_drawdown": DRAWDOWN_WINDOW - 1,
    "beta_60d": 60,
    "market_correlation_60d": 60,
    "avg_volume_20d": 19,
    "volume_change": 19,
}
# The slowest features need 60 earlier rows, so each stock loses its first 60 rows.
MIN_HISTORY_ROWS = max(FIRST_VALID_ROW.values())


# ---------------------------------------------------------------------------
# Small building blocks (each one is easy to explain on its own)
# ---------------------------------------------------------------------------
def _pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    """today / value `periods` rows ago - 1.   (e.g. 105 vs 100 -> 0.05)"""
    return series / series.shift(periods) - 1.0


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Relative Strength Index, simple-average version, scaled 0 to 100.

        change     = today's close - yesterday's close
        avg_gain   = average of the positive changes over the last `period` days (losses count as 0)
        avg_loss   = average of the negative changes over the last `period` days (as positive numbers)
        RS         = avg_gain / avg_loss
        RSI        = 100 - 100 / (1 + RS)

    High (above ~70) = mostly rising lately; low (below ~30) = mostly falling lately.
    Note: the textbook Wilder version smooths with a running average instead of a plain
    one, so its values differ slightly. We use the plain average because it is simpler
    to explain and to check by hand.

    Edge cases (no division by zero):
      no losses but some gains -> 100;  no gains and no losses (flat) -> 50.
    """
    change = close.diff()
    avg_gain = change.clip(lower=0).rolling(period).mean()
    avg_loss = (-change.clip(upper=0)).rolling(period).mean()

    rs = avg_gain / avg_loss.where(avg_loss > 0)  # NaN (not infinity) when avg_loss is 0
    rsi = 100.0 - 100.0 / (1.0 + rs)

    no_losses = np.where(avg_gain > 0, 100.0, 50.0)
    rsi = rsi.where(avg_loss > 0, no_losses)
    return rsi.where(avg_gain.notna())  # keep the warm-up period as missing


def _window_max_drawdown(prices: np.ndarray) -> float:
    """
    Worst fall from a peak inside one window of prices, as a decimal <= 0.

    Walk through the window in time order. At each day compare the price with the
    highest price SEEN SO FAR in the window (the running maximum). The most negative
    result is the max drawdown. A later high never affects an earlier day.
    """
    running_peak = np.maximum.accumulate(prices)
    return float((prices / running_peak - 1.0).min())


def _features_for_one_stock(stock: pd.DataFrame, market_close_by_date: pd.Series) -> pd.DataFrame:
    """All 13 features for ONE stock (rows already sorted by date)."""
    stock = stock.reset_index(drop=True)
    close = stock["close"]
    volume = stock["volume"]
    out = stock[ID_COLUMNS].copy()

    daily_return = _pct_change(close, 1)

    # --- 1. Returns: how much the price changed over the last 5 / 20 / 60 trading days ---
    # return_Nd = close today / close N days ago - 1
    out["return_5d"] = _pct_change(close, 5)
    out["return_20d"] = _pct_change(close, 20)
    out["return_60d"] = _pct_change(close, 60)

    # --- 2. Volatility: how bumpy the daily returns were (bigger = riskier) ---
    # Standard deviation of the last N daily returns, annualised by x sqrt(252).
    # (Daily noise adds up like a random walk, so a yearly figure is sqrt(252) times the daily one.)
    annualise = np.sqrt(TRADING_DAYS_PER_YEAR)
    out["volatility_20d"] = daily_return.rolling(20).std() * annualise
    out["volatility_60d"] = daily_return.rolling(60).std() * annualise

    # --- 3. Price position: is the price above (>1) or below (<1) its recent average? ---
    out["price_ma20_ratio"] = close / close.rolling(20).mean()
    out["price_ma50_ratio"] = close / close.rolling(50).mean()
    out["rsi_14"] = _rsi(close, RSI_PERIOD)

    # --- 4. Drawdown: worst fall from a peak within the last 60 trading days (0 = never fell, -0.2 = fell 20%) ---
    out["max_drawdown"] = close.rolling(DRAWDOWN_WINDOW).apply(_window_max_drawdown, raw=True)

    # --- 5. Market relationship (uses the market index) ---
    # Look up the index close on THIS stock's dates, then take the index return over the same
    # interval as the stock's return. If the index has no value for a date, the return is
    # missing (NaN); we never guess it.
    market_close = stock["date"].map(market_close_by_date)
    market_return = _pct_change(market_close, 1)

    # beta = covariance(stock, market) / variance(market): "if the market moves 1%, the stock
    # tends to move beta %". Both pieces use the same 60-day window and the same ddof, so they cancel.
    market_var = market_return.rolling(60).var()
    out["beta_60d"] = daily_return.rolling(60).cov(market_return) / market_var.where(market_var > 0)
    # correlation: from -1 (opposite) to +1 (moves in lockstep), ignoring size
    out["market_correlation_60d"] = daily_return.rolling(60).corr(market_return)

    # --- 6. Volume ---
    out["avg_volume_20d"] = volume.rolling(20).mean()
    # volume_change = today's volume relative to the 20-day average, minus 1.
    # 0 = normal day, 1.0 = twice the usual volume, -0.5 = half. If the average is 0
    # (e.g. volume was missing and Phase 1 set it to 0) the result is NaN, not infinity.
    out["volume_change"] = volume / out["avg_volume_20d"].where(out["avg_volume_20d"] > 0) - 1.0

    return out


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def build_features(stock_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Turn price tables into a feature table.

    stock_data:  Phase 1 table (needs date, symbol, close, volume), any number of symbols.
    market_data: Phase 1 table for ONE market index (needs date, close).
                 It is matched to the stocks by date, so row order does not matter.

    Returns one row per input stock-day with columns
        date, symbol, close + the 13 FEATURE_COLUMNS
    sorted by symbol then date. Early rows contain NaN because a 60-day feature cannot
    exist until 60 days of history do. We leave those NaN (no filling, no inventing).
    Use clean_feature_data() to drop them.
    """
    _check_stock_data(stock_data)
    market_close_by_date = _market_close_by_date(market_data)

    stocks = stock_data[["date", "symbol", "close", "volume"]].copy()
    stocks["date"] = pd.to_datetime(stocks["date"])
    stocks = stocks.sort_values(["symbol", "date"], kind="stable")

    # One stock at a time, so one stock's history can never bleed into another's.
    pieces = [_features_for_one_stock(group, market_close_by_date) for _, group in stocks.groupby("symbol", sort=True)]
    features = pd.concat(pieces, ignore_index=True)[OUTPUT_COLUMNS]

    # Safety net: a division by zero must show up as "missing", never as infinity.
    features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return features


def missing_value_report(features: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Count missing values.

    Returns (table, rows_with_any_missing). The table has one row per feature:
    how many rows are missing it and what percent of all rows that is.
    """
    missing = features[FEATURE_COLUMNS].isna().sum()
    table = pd.DataFrame({"missing_rows": missing, "missing_%": (missing / len(features) * 100).round(2)})
    rows_with_any_missing = int(features[FEATURE_COLUMNS].isna().any(axis=1).sum())
    return table, rows_with_any_missing


def clean_feature_data(features: pd.DataFrame) -> pd.DataFrame:
    """
    Drop every row that lacks the complete feature set.

    In practice that is the first 60 rows of each stock: 60-day features need 60 earlier
    days (return_60d needs the close from 60 days ago, volatility_60d / beta_60d need 60
    daily returns, which need 61 closes). We remove those rows rather than invent history.
    Nothing is filled in; the remaining rows are unchanged.
    """
    cleaned = features.dropna(subset=FEATURE_COLUMNS)
    return cleaned.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------
def _check_stock_data(stock_data: pd.DataFrame) -> None:
    missing = [c for c in ["date", "symbol", "close", "volume"] if c not in stock_data.columns]
    if missing:
        raise ValueError(f"stock_data is missing column(s): {missing}")
    if stock_data.duplicated(subset=["symbol", "date"]).any():
        raise ValueError("stock_data has duplicate (symbol, date) rows. Clean it with the Phase 1 loader first.")
    if stock_data["close"].isna().any():
        raise ValueError("stock_data has missing close prices. Clean it with the Phase 1 loader first.")


def _market_close_by_date(market_data: pd.DataFrame) -> pd.Series:
    """The market index close, indexed by date, ready to be matched to each stock's dates."""
    missing = [c for c in ["date", "close"] if c not in market_data.columns]
    if missing:
        raise ValueError(f"market_data is missing column(s): {missing}")
    if "symbol" in market_data.columns and market_data["symbol"].nunique() > 1:
        raise ValueError("market_data must contain a single market index, but it has several symbols.")
    market = market_data[["date", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"])
    if market["date"].duplicated().any():
        raise ValueError("market_data has duplicate dates.")
    return market.set_index("date")["close"].sort_index()


# ---------------------------------------------------------------------------
# Command-line demonstration
# ---------------------------------------------------------------------------
def print_demo(stock_data: pd.DataFrame, market_data: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Build the features, clean them, print a readable report, return the cleaned table."""
    features = build_features(stock_data, market_data)
    table, rows_with_nan = missing_value_report(features)
    final = clean_feature_data(features)
    removed = len(features) - len(final)

    print("=== FEATURE ENGINEERING ===")
    print()
    if "synthetic" in source.lower():
        print("NOTE: synthetic development data (random numbers, NOT real prices).")
        print("These features describe simulated history only; they say nothing about real stocks.")
        print()
    print(f"Input rows:  {len(stock_data):,}")
    print(f"Output rows: {len(features):,}  (one per stock-day, before cleaning)")
    print()
    print(f"Stocks:     {features['symbol'].nunique()}")
    print(f"Date range: {features['date'].min().date()}  to  {features['date'].max().date()}")
    print()
    print("Features created:")
    for name in FEATURE_COLUMNS:
        print(f"- {name}")
    print()
    print("Missing values before cleaning (expected: warm-up rows at the start of each stock):")
    print(table.to_string())
    print(f"Rows with at least one missing feature: {rows_with_nan:,}")
    print()
    print("Rows removed due to insufficient history:")
    n_stocks = features["symbol"].nunique()
    print(f"{removed:,}  (about {MIN_HISTORY_ROWS} per stock x {n_stocks} stocks = {MIN_HISTORY_ROWS * n_stocks:,})")
    lost = sorted(set(features["symbol"]) - set(final["symbol"]))
    if lost:
        print(f"WARNING: no usable rows left for: {lost} (less than {MIN_HISTORY_ROWS + 1} rows of history)")
    print()
    print(f"Final feature rows: {len(final):,}")
    print(f"Missing values after cleaning: {int(final[FEATURE_COLUMNS].isna().sum().sum())}")
    print(f"Infinite values after cleaning: {int(np.isinf(final[FEATURE_COLUMNS]).sum().sum())}")
    print()
    print("Feature summary:")
    summary = final[FEATURE_COLUMNS].agg(["mean", "std", "min", "max"]).T
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Build, clean and summarise the Phase 2 features.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    config = load_config(config_path)
    cfg = config["data"]
    window = {"start": cfg.get("start_date"), "end": cfg.get("end_date")}

    provider = get_provider(config, base_dir=base_dir)
    stock_data = provider.get_historical_data(symbols=cfg.get("symbols"), **window)

    # The market index comes from its own file for now (Phase 10 will source it from the provider too).
    market_provider = LocalDataProvider(base_dir / cfg["market_index_file"])
    market_data = market_provider.get_historical_data(**window)

    print_demo(stock_data, market_data, source=str(getattr(provider, "path", "")))


if __name__ == "__main__":
    main()
