"""
Phase 1: data loading.

What this file does, in plain words
-----------------------------------
1. Defines ONE standard table format for price data ("the OHLCV format"):

       date | symbol | open | high | low | close | volume

2. Defines a `DataProvider` interface: "something that can give me that table".
   - `LocalDataProvider` reads a CSV/Parquet file (used now).
   - An Angel One provider is planned for Phase 8E. It will return the SAME
     table, so no machine-learning code has to change when we switch. Until
     then `get_provider` refuses to build it (it never falls back to local data).

3. Cleans and validates the table so later phases can trust it.

Run it:   python -m src.data_loader
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The standard columns every data source must produce, in this order.
REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["volume"]


class DataValidationError(ValueError):
    """Raised when the data is broken in a way we refuse to guess our way around."""


class ProviderConfigurationError(ValueError):
    """Raised when the requested data provider does not exist (for example DATA_PROVIDER=yahoo)."""


class ProviderNotImplementedError(NotImplementedError):
    """Raised when a provider is known but not built yet (Angel One, until Phase 8E). Never replaced by local data."""


class DataUnavailableError(RuntimeError):
    """Raised by `DataProvider.check_available()` when a provider cannot currently supply data."""


# Provider names accepted by `get_provider`. "local" is the default. "angel_one" is reserved for Phase 8E.
PROVIDER_LOCAL = "local"
PROVIDER_ANGEL_ONE = "angel_one"
KNOWN_PROVIDERS = (PROVIDER_LOCAL, PROVIDER_ANGEL_ONE)
_PROVIDER_ALIASES = {"angelone": PROVIDER_ANGEL_ONE}  # the spelling config.yaml used before Phase 8D


def normalize_provider_name(name: object) -> str:
    """'  Local ' -> 'local', 'angelone' -> 'angel_one'. An empty value means "use the default" and returns ''."""
    text = str(name or "").strip().lower()
    return _PROVIDER_ALIASES.get(text, text)


# ---------------------------------------------------------------------------
# 1. Cleaning
# ---------------------------------------------------------------------------
def clean_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Turn a raw price table into a clean one.

    Returns (clean_dataframe, report). The report counts what we changed, so
    nothing happens silently.

    Rules (kept deliberately simple):
      * Missing columns, unparseable dates or missing symbols -> error (structural problem).
      * Text in a number column (e.g. "abc") becomes "missing".
      * Duplicate (symbol, date) rows -> keep the last one.
      * Row with NO close price -> dropped. We never invent a close price.
      * Missing open/high/low but close exists -> rebuilt from open/close:
            open = close,  high = max(open, close),  low = min(open, close)
      * Missing volume -> 0.
      * Result sorted by symbol, then date.

    LEAKAGE NOTE: we do NOT forward-fill or back-fill prices. Back-filling would
    copy FUTURE prices into the past, which is a classic leakage mistake.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required column(s): {missing_cols}. Expected: {REQUIRED_COLUMNS}")

    out = df[REQUIRED_COLUMNS].copy()
    report = {"rows_in": len(out)}

    # --- dates: must all parse. Daily data, so we drop any time-of-day. ---
    parsed = pd.to_datetime(out["date"], errors="coerce")
    bad_dates = int(parsed.isna().sum())
    if bad_dates:
        raise DataValidationError(f"{bad_dates} row(s) have a missing or unparseable date.")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    out["date"] = parsed.dt.normalize()

    # --- symbols: must exist; make them consistent ("  tcs " -> "TCS"). ---
    if out["symbol"].isna().any():
        raise DataValidationError(f"{int(out['symbol'].isna().sum())} row(s) have a missing symbol.")
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()

    # --- numbers: anything that is not a number becomes NaN ("missing"). ---
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    # --- sort, then remove duplicate (symbol, date) rows ---
    out = out.sort_values(["symbol", "date"], kind="stable")
    dupes = out.duplicated(subset=["symbol", "date"], keep="last")
    report["duplicates_removed"] = int(dupes.sum())
    out = out[~dupes]

    # --- no close price -> drop the row ---
    no_close = out["close"].isna()
    report["rows_dropped_no_close"] = int(no_close.sum())
    out = out[~no_close].copy()

    # --- rebuild missing open/high/low from the prices we do have ---
    n_missing_ohl = int(out[["open", "high", "low"]].isna().sum().sum())
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out[["open", "close"]].max(axis=1))
    out["low"] = out["low"].fillna(out[["open", "close"]].min(axis=1))
    report["open_high_low_filled"] = n_missing_ohl

    # --- missing volume -> 0 ---
    report["volume_filled_with_zero"] = int(out["volume"].isna().sum())
    out["volume"] = out["volume"].fillna(0.0)

    out = out.reset_index(drop=True)
    report["rows_out"] = len(out)
    return out, report


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------
def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Check a CLEANED table. Raises DataValidationError listing every problem found.

    Checks:
      * all required columns present, no missing values
      * date column is a real datetime
      * no duplicate (symbol, date) rows, table sorted by symbol then date
      * prices > 0 and volume >= 0
      * impossible candles: high < low, high below open/close, low above open/close
    """
    problems: list[str] = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required column(s): {missing_cols}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        problems.append("'date' column is not a datetime type")

    n_nan = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    if n_nan:
        problems.append(f"{n_nan} missing value(s) remain")

    if df.duplicated(subset=["symbol", "date"]).any():
        problems.append("duplicate (symbol, date) rows exist")

    if not df.sort_values(["symbol", "date"], kind="stable").index.equals(df.index):
        problems.append("rows are not sorted by symbol then date")

    # Only run the numeric checks if the numbers are really numbers.
    if all(pd.api.types.is_numeric_dtype(df[c]) for c in NUMERIC_COLUMNS):
        checks = {
            "non-positive price (open/high/low/close <= 0)": (df[PRICE_COLUMNS] <= 0).any(axis=1),
            "negative volume": df["volume"] < 0,
            "high < low": df["high"] < df["low"],
            "high below open or close": df["high"] < df[["open", "close"]].max(axis=1),
            "low above open or close": df["low"] > df[["open", "close"]].min(axis=1),
        }
        for name, bad_mask in checks.items():
            n_bad = int(bad_mask.sum())
            if n_bad:
                example = df.loc[bad_mask, ["symbol", "date"]].iloc[0]
                problems.append(f"{n_bad} row(s) with {name} (first: {example['symbol']} on {example['date'].date()})")
    else:
        problems.append("price/volume columns are not all numeric")

    if problems:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# 3. File reading
# ---------------------------------------------------------------------------
def read_price_file(path: str | Path) -> pd.DataFrame:
    """Read a .csv or .parquet file into a DataFrame (no cleaning yet)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Price file not found: {path}\n"
            "Create the development data with:  python -m src.sample_data\n"
            "or place your own file there (columns: date, symbol, open, high, low, close, volume)."
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Use .csv or .parquet.")


# ---------------------------------------------------------------------------
# 4. Data providers (the swap-in point for Angel One)
# ---------------------------------------------------------------------------
class DataProvider(ABC):
    """
    Anything that can supply historical prices in the standard OHLCV format.

    The rest of the project only ever calls `get_historical_data(...)`. It does
    not know or care where the numbers came from.

    THE PROVIDER CONTRACT (tests/test_data_provider_contract.py checks every item for a provider):
      * returns exactly REQUIRED_COLUMNS, in that order
      * date: datetime64 with no time of day and no timezone; symbol: upper-case text
      * open/high/low/close/volume: finite float64 (no missing values)
      * prices > 0, volume >= 0, high >= max(open, close, low), low <= min(open, close)
      * sorted by symbol then date, one row per (symbol, date)
      * symbols are matched case-insensitively; an unknown symbol raises ValueError
      * start/end are inclusive; start after end, or nothing left in the window, raises ValueError
      * broken data raises DataValidationError (it is never silently repaired); a row with no close price
        is absent from the result (a price is never invented)
    """

    # Short identifier reported by the API (`provider` in /api/dataset and /api/health).
    name: str = "unknown"
    # True only for generated development data. Real sources keep the default.
    is_synthetic: bool = False

    def check_available(self) -> None:
        """Cheap readiness check used by /api/health. Raise DataUnavailableError if data cannot be supplied. Default: ready."""

    @abstractmethod
    def get_historical_data(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Return a clean, validated DataFrame with columns
        date, symbol, open, high, low, close, volume (sorted by symbol, date).

        symbols: list of symbols, or None for all available.
        start/end: inclusive date limits like "2023-01-31", or None.

        LEAKAGE NOTE: `end` is how later phases will cut the data off at the end
        of the training period, so nothing after that date can leak in.
        """


class LocalDataProvider(DataProvider):
    """Reads prices from a local CSV/Parquet file."""

    name = PROVIDER_LOCAL

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.last_cleaning_report: dict = {}

    @property
    def is_synthetic(self) -> bool:  # type: ignore[override]
        """The development files are generated; they are recognised by 'synthetic' in the file name."""
        return "synthetic" in self.path.name.lower()

    def check_available(self) -> None:
        if not self.path.is_file():
            raise DataUnavailableError("The configured price file was not found.")

    def get_historical_data(self, symbols=None, start=None, end=None) -> pd.DataFrame:
        raw = read_price_file(self.path)
        df, self.last_cleaning_report = clean_ohlcv(raw)
        validate_ohlcv(df)

        if symbols is not None:
            wanted = [s.strip().upper() for s in symbols]
            unknown = sorted(set(wanted) - set(df["symbol"]))
            if unknown:
                raise ValueError(f"Symbols not found in {self.path.name}: {unknown}")
            df = df[df["symbol"].isin(wanted)]

        start_ts = pd.Timestamp(start) if start else None
        end_ts = pd.Timestamp(end) if end else None
        if start_ts is not None and end_ts is not None and start_ts > end_ts:
            raise ValueError(f"start ({start}) is after end ({end}).")
        if start_ts is not None:
            df = df[df["date"] >= start_ts]
        if end_ts is not None:
            df = df[df["date"] <= end_ts]

        if df.empty:
            raise ValueError("No data left after applying the symbol/date filters.")
        return df.reset_index(drop=True)


def get_provider(
    config: dict,
    base_dir: str | Path = ".",
    provider: str | None = None,
    data_path: str | Path | None = None,
) -> DataProvider:
    """
    Build the data provider (the single provider-selection point for the whole project).

    provider   overrides config.yaml's data.provider (the API passes the DATA_PROVIDER environment variable here).
               None or empty means "use config.yaml", which defaults to "local".
    data_path  overrides config.yaml's data.prices_file for the local provider (the API passes DATA_PATH). A relative
               path is resolved against base_dir (the project root), an absolute path is used as given.

    There is no silent fallback: an unknown name raises ProviderConfigurationError, and "angel_one" raises
    ProviderNotImplementedError until Phase 8E. Neither ever returns local data instead.
    """
    name = normalize_provider_name(provider) or normalize_provider_name(config["data"].get("provider")) or PROVIDER_LOCAL
    if name == PROVIDER_LOCAL:
        chosen = Path(data_path) if data_path else Path(config["data"]["prices_file"])
        return LocalDataProvider(chosen if chosen.is_absolute() else Path(base_dir) / chosen)
    if name == PROVIDER_ANGEL_ONE:
        raise ProviderNotImplementedError(
            "The Angel One data provider is not implemented yet (planned for Phase 8E). "
            "Set DATA_PROVIDER=local, or leave it unset, to use the local dataset."
        )
    shown = name if len(name) <= 40 else name[:40] + "..."
    raise ProviderConfigurationError(f"Unknown data provider '{shown}'. Supported values: {', '.join(KNOWN_PROVIDERS)}.")


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 5. Basic statistics
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per symbol: how much data, price range, and simple return/risk numbers.

    These are DESCRIPTIVE only. They are not used to fit any model, so there is
    no leakage concern here.
    """
    d = df.sort_values(["symbol", "date"])
    d = d.assign(daily_return=d.groupby("symbol")["close"].pct_change())
    g = d.groupby("symbol")
    summary = pd.DataFrame(
        {
            "rows": g.size(),
            "first_date": g["date"].min().dt.date,
            "last_date": g["date"].max().dt.date,
            "min_close": g["close"].min(),
            "max_close": g["close"].max(),
            "avg_daily_return_%": g["daily_return"].mean() * 100,
            "annual_volatility_%": g["daily_return"].std() * np.sqrt(252) * 100,  # 252 trading days/year
            "avg_volume": g["volume"].mean(),
        }
    )
    return summary.round(2)


def print_report(df: pd.DataFrame, cleaning_report: dict, source: str) -> None:
    print("=" * 72)
    print("DATA SUMMARY")
    print("=" * 72)
    if "synthetic" in source.lower():
        print("NOTE: this is SYNTHETIC development data (random numbers, NOT real prices).")
    print(f"Source file : {source}")
    print(f"Symbols     : {df['symbol'].nunique()}  ({', '.join(sorted(df['symbol'].unique()))})")
    print(f"Rows        : {len(df):,}")
    print(f"Date range  : {df['date'].min().date()}  to  {df['date'].max().date()}")
    print(f"Latest date : {df['date'].max().date()}")
    print()
    print("Cleaning report (what the loader changed):")
    for key, value in cleaning_report.items():
        print(f"  {key:<26}{value}")
    print()
    print("Per-symbol statistics:")
    print(summarise(df).to_string())
    print()
    print("Validation: PASSED (columns, dates, no missing values, no impossible OHLC, sorted).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load, validate and summarise the price data.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    provider = get_provider(config, base_dir=config_path.parent)

    cfg = config["data"]
    df = provider.get_historical_data(symbols=cfg.get("symbols"), start=cfg.get("start_date"), end=cfg.get("end_date"))
    report = getattr(provider, "last_cleaning_report", {})
    print_report(df, report, source=str(getattr(provider, "path", cfg["provider"])))


if __name__ == "__main__":
    main()
