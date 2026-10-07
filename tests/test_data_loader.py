"""Tests for Phase 1 (src/data_loader.py and src/sample_data.py). Run: pytest -q"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    REQUIRED_COLUMNS,
    DataValidationError,
    LocalDataProvider,
    clean_ohlcv,
    get_provider,
    load_config,
    summarise,
    validate_ohlcv,
)
from src.sample_data import DEV_STOCKS, generate

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------- small hand-made table used by most tests ----------
@pytest.fixture
def raw_df():
    """Two symbols x 3 days. Deliberately unsorted, with messy symbol text."""
    return pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-01", "2024-01-03"],
            "symbol": [" tcs", "TCS ", "tcs", "INFY", "infy", "INFY"],
            "open": [102, 100, 101, 50, 49, 51],
            "high": [104, 103, 105, 52, 51, 53],
            "low": [100, 99, 100, 49, 48, 50],
            "close": [103, 101, 104, 51, 50, 52],
            "volume": [1000, 900, 1100, 500, 400, 600],
        }
    )


@pytest.fixture
def clean_df(raw_df):
    df, _ = clean_ohlcv(raw_df)
    return df


# ======================= DATA: required columns =======================
def test_missing_required_column_raises(raw_df):
    with pytest.raises(DataValidationError, match="volume"):
        clean_ohlcv(raw_df.drop(columns=["volume"]))


def test_output_has_exactly_the_required_columns_in_order(clean_df):
    assert list(clean_df.columns) == REQUIRED_COLUMNS


# ======================= DATA: dates =======================
def test_dates_become_datetime_with_no_time_of_day(clean_df):
    assert pd.api.types.is_datetime64_any_dtype(clean_df["date"])
    assert (clean_df["date"] == clean_df["date"].dt.normalize()).all()
    assert clean_df["date"].min() == pd.Timestamp("2024-01-01")


def test_timezone_aware_dates_are_made_naive(raw_df):
    raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.tz_localize("Asia/Kolkata")
    df, _ = clean_ohlcv(raw_df)
    assert df["date"].dt.tz is None


@pytest.mark.filterwarnings("ignore:Could not infer format")  # pandas warns when the very first date is junk
def test_unparseable_date_raises(raw_df):
    raw_df.loc[0, "date"] = "not-a-date"
    with pytest.raises(DataValidationError, match="date"):
        clean_ohlcv(raw_df)


# ======================= DATA: sorting, symbols, duplicates =======================
def test_sorted_by_symbol_then_date(clean_df):
    assert clean_df.sort_values(["symbol", "date"]).index.equals(clean_df.index)
    assert list(clean_df["symbol"].unique()) == ["INFY", "TCS"]


def test_symbols_are_stripped_and_uppercased(clean_df):
    assert set(clean_df["symbol"]) == {"TCS", "INFY"}


def test_duplicate_rows_are_removed_keeping_the_last(raw_df):
    dup = raw_df.iloc[[1]].copy()  # TCS on 2024-01-01 again, with a different close
    dup["close"] = 100.5
    df, report = clean_ohlcv(pd.concat([raw_df, dup], ignore_index=True))
    assert report["duplicates_removed"] == 1
    row = df[(df["symbol"] == "TCS") & (df["date"] == "2024-01-01")]
    assert len(row) == 1 and row["close"].iloc[0] == 100.5


# ======================= DATA: missing values =======================
def test_row_without_close_is_dropped_not_invented(raw_df):
    raw_df.loc[2, "close"] = np.nan
    df, report = clean_ohlcv(raw_df)
    assert report["rows_dropped_no_close"] == 1
    assert len(df) == len(raw_df) - 1


def test_missing_open_high_low_are_rebuilt_validly(raw_df):
    raw_df.loc[0, ["open", "high", "low"]] = np.nan
    df, report = clean_ohlcv(raw_df)
    assert report["open_high_low_filled"] == 3
    validate_ohlcv(df)  # must still be a legal candle


def test_missing_volume_becomes_zero(raw_df):
    raw_df.loc[0, "volume"] = np.nan
    df, report = clean_ohlcv(raw_df)
    assert report["volume_filled_with_zero"] == 1
    assert df["volume"].isna().sum() == 0


def test_text_in_price_column_is_treated_as_missing(raw_df):
    raw_df["close"] = raw_df["close"].astype(object)
    raw_df.loc[1, "close"] = "abc"
    _, report = clean_ohlcv(raw_df)
    assert report["rows_dropped_no_close"] == 1


# ======================= DATA: impossible OHLC values =======================
def test_valid_data_passes_validation(clean_df):
    validate_ohlcv(clean_df)


@pytest.mark.parametrize(
    "column, value, expected_message",
    [
        ("high", 40, "high < low"),  # row 0 is INFY: low is 48, so a high of 40 is impossible
        ("close", 200, "high below open or close"),  # close above high
        ("open", 1, "low above open or close"),  # open below low
        ("close", -5, "non-positive price"),
        ("volume", -1, "negative volume"),
    ],
)
def test_validation_catches_impossible_values(clean_df, column, value, expected_message):
    bad = clean_df.copy()
    bad.loc[0, column] = value
    with pytest.raises(DataValidationError, match=expected_message):
        validate_ohlcv(bad)


def test_validation_catches_duplicates_and_unsorted(clean_df):
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_ohlcv(pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True))
    with pytest.raises(DataValidationError, match="not sorted"):
        validate_ohlcv(clean_df.iloc[::-1])


# ======================= PROVIDER =======================
@pytest.fixture
def csv_path(tmp_path, raw_df):
    path = tmp_path / "prices.csv"
    raw_df.to_csv(path, index=False)
    return path


def test_local_provider_reads_csv(csv_path):
    df = LocalDataProvider(csv_path).get_historical_data()
    assert len(df) == 6 and list(df.columns) == REQUIRED_COLUMNS


def test_local_provider_reads_parquet(tmp_path, raw_df):
    pytest.importorskip("pyarrow")
    path = tmp_path / "prices.parquet"
    raw_df.to_parquet(path)
    assert len(LocalDataProvider(path).get_historical_data()) == 6


def test_provider_filters_symbols_and_dates(csv_path):
    provider = LocalDataProvider(csv_path)
    only_tcs = provider.get_historical_data(symbols=["tcs"])
    assert set(only_tcs["symbol"]) == {"TCS"}
    window = provider.get_historical_data(start="2024-01-02", end="2024-01-02")  # inclusive on both ends
    assert set(window["date"]) == {pd.Timestamp("2024-01-02")}


def test_provider_errors_are_clear(csv_path, tmp_path):
    provider = LocalDataProvider(csv_path)
    with pytest.raises(ValueError, match="not found"):
        provider.get_historical_data(symbols=["NOPE"])
    with pytest.raises(ValueError, match="after end"):
        provider.get_historical_data(start="2024-02-01", end="2024-01-01")
    with pytest.raises(FileNotFoundError):
        LocalDataProvider(tmp_path / "missing.csv").get_historical_data()
    with pytest.raises(ValueError, match="Unsupported"):
        (tmp_path / "x.txt").write_text("hi")
        LocalDataProvider(tmp_path / "x.txt").get_historical_data()


def test_provider_factory():
    assert isinstance(get_provider({"data": {"provider": "local", "prices_file": "a.csv"}}), LocalDataProvider)
    # Phase 8D: Angel One moved from "Phase 10" to Phase 8E, and "angelone" is now an alias of "angel_one".
    with pytest.raises(NotImplementedError, match="Phase 8E"):
        get_provider({"data": {"provider": "angelone"}})
    with pytest.raises(ValueError, match="Unknown"):
        get_provider({"data": {"provider": "yahoo"}})


# ======================= CONFIG + DEV DATASET =======================
def test_config_has_the_keys_we_need():
    config = load_config(PROJECT_ROOT / "config.yaml")
    for key in ["provider", "prices_file", "market_index_file", "symbols", "start_date", "end_date"]:
        assert key in config["data"]


def test_dev_dataset_loads_and_is_valid():
    config = load_config(PROJECT_ROOT / "config.yaml")
    df = get_provider(config, PROJECT_ROOT).get_historical_data()
    assert df["symbol"].nunique() >= 10
    assert df.isna().sum().sum() == 0
    assert df["date"].min() < df["date"].max()
    validate_ohlcv(df)


def test_dev_data_generator_is_valid_and_repeatable():
    prices_a, index_a = generate(seed=7, start="2020-01-01", end="2020-12-31")
    prices_b, _ = generate(seed=7, start="2020-01-01", end="2020-12-31")
    prices_c, _ = generate(seed=8, start="2020-01-01", end="2020-12-31")
    pd.testing.assert_frame_equal(prices_a, prices_b)  # same seed -> identical data
    assert not prices_a["close"].equals(prices_c["close"])  # different seed -> different data
    assert set(prices_a["symbol"]) == set(DEV_STOCKS)
    clean, _ = clean_ohlcv(prices_a)
    validate_ohlcv(clean)
    clean_index, _ = clean_ohlcv(index_a)
    validate_ohlcv(clean_index)


# ======================= STATISTICS =======================
def test_summary_has_one_row_per_symbol_with_correct_counts(clean_df):
    s = summarise(clean_df)
    assert list(s.index) == ["INFY", "TCS"]
    assert (s["rows"] == 3).all()
    assert s.loc["TCS", "min_close"] == 101 and s.loc["TCS", "max_close"] == 104
    # In date order TCS closes 101 (Jan 1) -> 104 (Jan 2) -> 103 (Jan 3).
    # Average of the two daily returns, in percent:
    expected = np.mean([104 / 101 - 1, 103 / 104 - 1]) * 100
    assert s.loc["TCS", "avg_daily_return_%"] == pytest.approx(expected, abs=0.01)
