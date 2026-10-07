"""
Creates the SYNTHETIC development dataset.

IMPORTANT: these are random numbers produced by a small simulation. They are NOT
real market prices. The ticker names (RELIANCE, TCS, ...) are only labels so the
project looks familiar while we build it. Nothing computed from this data says
anything about the real stocks. Real data arrives in Phase 10 (Angel One).

Why simulate instead of downloading? (1) it needs no account or internet,
(2) we never present invented numbers as real, (3) we control the structure
(some stocks move together, some are riskier) so PCA/LDA have something to find.

How the simulation works
------------------------
Every day:
  market return  = a random number; the market has calm and stressed periods
  sector shock   = a random number shared by stocks of the same sector
  stock return   = drift + beta * market return + sector shock + own noise
So stocks in the same sector move together, and high-beta stocks swing more
with the market. Prices, highs/lows and volumes are then built from that.

Run it:   python -m src.sample_data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# symbol: (sector, start price, beta, yearly drift, yearly own-noise vol, typical daily volume)
# These numbers are invented "personalities". They are not estimates of real stocks.
DEV_STOCKS = {
    "RELIANCE": ("Energy", 1000.0, 1.05, 0.10, 0.18, 6_000_000),
    "TCS": ("IT", 2000.0, 0.80, 0.12, 0.15, 2_000_000),
    "INFY": ("IT", 900.0, 0.90, 0.11, 0.17, 5_000_000),
    "HDFCBANK": ("Banks", 1100.0, 1.00, 0.12, 0.14, 7_000_000),
    "ICICIBANK": ("Banks", 300.0, 1.15, 0.14, 0.18, 9_000_000),
    "SBIN": ("Banks", 280.0, 1.35, 0.13, 0.26, 15_000_000),
    "AXISBANK": ("Banks", 550.0, 1.25, 0.11, 0.24, 8_000_000),
    "ITC": ("FMCG", 250.0, 0.65, 0.09, 0.14, 12_000_000),
    "LT": ("Industrials", 1300.0, 1.10, 0.12, 0.19, 2_500_000),
    "MARUTI": ("Auto", 7000.0, 0.95, 0.08, 0.20, 400_000),
}

INDEX_SYMBOL = "MARKET_INDEX"
TRADING_DAYS = 252


def _simulate_market(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """Daily market log-returns with calm and stressed regimes (volatility switches)."""
    calm_vol, stressed_vol = 0.009, 0.022  # daily
    p_calm_to_stress, p_stress_to_calm = 0.01, 0.06
    stressed = False
    vol = np.empty(n_days)
    for t in range(n_days):
        stressed = (rng.random() < p_calm_to_stress) if not stressed else (rng.random() >= p_stress_to_calm)
        vol[t] = stressed_vol if stressed else calm_vol
    return 0.12 / TRADING_DAYS + vol * rng.standard_normal(n_days)


def _build_ohlcv(rng, dates, symbol, start_price, log_returns, base_volume) -> pd.DataFrame:
    """Turn a series of daily log-returns into open/high/low/close/volume rows."""
    close = start_price * np.exp(np.cumsum(log_returns))
    prev_close = np.concatenate([[start_price], close[:-1]])
    open_ = prev_close * np.exp(0.002 * rng.standard_normal(len(close)))  # small overnight gap
    high = np.maximum(open_, close) * np.exp(np.abs(0.004 * rng.standard_normal(len(close))))
    low = np.minimum(open_, close) * np.exp(-np.abs(0.004 * rng.standard_normal(len(close))))
    # Busier days (bigger moves) trade more shares.
    volume = base_volume * np.exp(0.35 * rng.standard_normal(len(close))) * (1 + 15 * np.abs(log_returns))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": symbol,
            "open": open_.round(2),
            "high": high.round(2),  # rounding keeps high >= open/close and low <= open/close
            "low": low.round(2),
            "close": close.round(2),
            "volume": volume.round().astype("int64"),
        }
    )


def generate(seed: int = 42, start: str = "2018-01-01", end: str = "2025-12-31") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (stock_prices, market_index) as two OHLCV tables. Same seed -> same data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)  # Mon-Fri. NSE holidays are ignored in this synthetic data.
    n = len(dates)

    market = _simulate_market(rng, n)
    sectors = sorted({v[0] for v in DEV_STOCKS.values()})
    sector_shock = {s: 0.006 * rng.standard_normal(n) for s in sectors}

    stock_frames = []
    for symbol, (sector, price0, beta, drift, idio_vol, volume) in DEV_STOCKS.items():
        own_noise = idio_vol / np.sqrt(TRADING_DAYS) * rng.standard_normal(n)
        log_ret = drift / TRADING_DAYS + beta * (market - 0.12 / TRADING_DAYS) + sector_shock[sector] + own_noise
        stock_frames.append(_build_ohlcv(rng, dates, symbol, price0, log_ret, volume))

    index_df = _build_ohlcv(rng, dates, INDEX_SYMBOL, 10_000.0, market, 0)
    index_df["volume"] = 0  # an index itself has no traded volume
    return pd.concat(stock_frames, ignore_index=True), index_df


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    prices, index_df = generate()
    prices_path = raw_dir / "dev_prices_synthetic.csv"
    index_path = raw_dir / "dev_market_index_synthetic.csv"
    prices.to_csv(prices_path, index=False, date_format="%Y-%m-%d")
    index_df.to_csv(index_path, index=False, date_format="%Y-%m-%d")

    print("Wrote SYNTHETIC development data (random numbers, not real prices):")
    print(f"  {prices_path}  ({len(prices):,} rows, {prices['symbol'].nunique()} symbols)")
    print(f"  {index_path}  ({len(index_df):,} rows)")


if __name__ == "__main__":
    main()
