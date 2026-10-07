"""
The DataProvider contract (Phase 8D).

Every data provider, now and in Phase 8E, must satisfy these tests. `ProviderContract` is written against the abstract
`DataProvider` interface only. To put a new provider under contract, subclass it and supply two fixtures:

    provider                 a provider serving valid data (at least 5 symbols, a few hundred trading days)
    provider_from_frame      a factory: DataFrame -> provider serving exactly that raw data. It is how the contract feeds
                             in broken data to check the provider refuses it. A provider that cannot be fed arbitrary
                             raw rows (a live API) supplies a fake transport underneath instead.

The local provider is the only one under contract today (`TestLocalProviderContract`).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    REQUIRED_COLUMNS,
    DataProvider,
    DataUnavailableError,
    DataValidationError,
    LocalDataProvider,
)

ROOT = Path(__file__).resolve().parent.parent
DEV_PRICES = ROOT / "data" / "raw" / "dev_prices_synthetic.csv"


def raw_frame() -> pd.DataFrame:
    """A small clean table in the standard format: 3 symbols x 10 business days."""
    days = pd.bdate_range("2023-01-02", periods=10)
    rows = []
    for k, symbol in enumerate(["AAA", "BBB", "CCC"]):
        for i, day in enumerate(days):
            close = 100.0 + 10 * k + i
            rows.append({"date": day, "symbol": symbol, "open": close - 0.5, "high": close + 1.0, "low": close - 1.0,
                         "close": close, "volume": 1000 + 10 * i})
    return pd.DataFrame(rows)


class ProviderContract:
    """Subclass and provide the `provider` and `provider_from_frame` fixtures."""

    # ------------------------------------------------------------------ identity
    def test_is_a_data_provider_with_a_name(self, provider):
        assert isinstance(provider, DataProvider)
        assert isinstance(provider.name, str) and provider.name and provider.name != "unknown"
        assert isinstance(provider.is_synthetic, bool)

    def test_a_ready_provider_passes_its_availability_check(self, provider):
        provider.check_available()  # must not raise

    # ------------------------------------------------------------------ shape and types
    def test_returns_exactly_the_standard_columns_in_order(self, provider):
        assert list(provider.get_historical_data().columns) == REQUIRED_COLUMNS

    def test_date_is_a_plain_daily_datetime(self, provider):
        dates = provider.get_historical_data()["date"]
        assert pd.api.types.is_datetime64_any_dtype(dates)
        assert getattr(dates.dt, "tz", None) is None
        assert (dates == dates.dt.normalize()).all()  # no time of day

    def test_symbol_is_stripped_upper_case_text(self, provider):
        symbols = provider.get_historical_data()["symbol"]
        assert symbols.map(lambda s: isinstance(s, str)).all()
        assert (symbols == symbols.str.strip().str.upper()).all()
        assert (symbols != "").all()

    def test_prices_and_volume_are_finite_floats(self, provider):
        df = provider.get_historical_data()
        for column in ["open", "high", "low", "close", "volume"]:
            assert df[column].dtype == np.float64, column
            assert np.isfinite(df[column]).all(), column

    # ------------------------------------------------------------------ valid values
    def test_no_impossible_ohlc_values(self, provider):
        df = provider.get_historical_data()
        assert (df[["open", "high", "low", "close"]] > 0).all().all()
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()

    def test_no_negative_volume(self, provider):
        assert (provider.get_historical_data()["volume"] >= 0).all()

    # ------------------------------------------------------------------ ordering
    def test_sorted_by_symbol_then_date_with_no_duplicates(self, provider):
        df = provider.get_historical_data()
        assert df.sort_values(["symbol", "date"], kind="stable").index.equals(df.index)
        assert not df.duplicated(subset=["symbol", "date"]).any()

    def test_the_index_is_a_clean_range(self, provider):
        df = provider.get_historical_data()
        assert df.index.equals(pd.RangeIndex(len(df)))

    def test_two_calls_give_identical_tables(self, provider):
        pd.testing.assert_frame_equal(provider.get_historical_data(), provider.get_historical_data())

    # ------------------------------------------------------------------ symbols
    def test_symbol_filter_returns_only_those_symbols_case_insensitively(self, provider):
        everything = provider.get_historical_data()
        picked = sorted(everything["symbol"].unique())[:2]
        df = provider.get_historical_data(symbols=[picked[0].lower(), f" {picked[1]} "])
        assert sorted(df["symbol"].unique()) == picked

    def test_an_unknown_symbol_raises_and_names_it(self, provider):
        with pytest.raises(ValueError, match="NOSUCHSTOCK"):
            provider.get_historical_data(symbols=["nosuchstock"])

    def test_one_unknown_symbol_among_known_ones_still_raises(self, provider):
        known = provider.get_historical_data()["symbol"].iloc[0]
        with pytest.raises(ValueError, match="NOSUCHSTOCK"):
            provider.get_historical_data(symbols=[known, "NOSUCHSTOCK"])

    # ------------------------------------------------------------------ dates
    def test_start_and_end_are_inclusive(self, provider):
        everything = provider.get_historical_data()
        days = sorted(everything["date"].unique())
        start, end = pd.Timestamp(days[3]), pd.Timestamp(days[9])
        df = provider.get_historical_data(start=str(start.date()), end=str(end.date()))
        assert df["date"].min() == start and df["date"].max() == end

    def test_end_cuts_the_data_so_nothing_later_is_returned(self, provider):
        days = sorted(provider.get_historical_data()["date"].unique())
        end = pd.Timestamp(days[len(days) // 2])
        assert provider.get_historical_data(end=str(end.date()))["date"].max() <= end

    def test_start_after_end_raises(self, provider):
        with pytest.raises(ValueError, match="after"):
            provider.get_historical_data(start="2024-01-01", end="2023-01-01")

    def test_a_window_with_no_data_raises_instead_of_returning_nothing(self, provider):
        with pytest.raises(ValueError, match="No data"):
            provider.get_historical_data(start="2090-01-01")

    # ------------------------------------------------------------------ missing data is explicit, never invented
    def test_a_row_with_no_close_is_absent_not_invented(self, provider_from_frame):
        raw = raw_frame()
        gone = raw[(raw["symbol"] == "BBB")].iloc[4]
        raw.loc[gone.name, "close"] = np.nan
        df = provider_from_frame(raw).get_historical_data()
        assert len(df) == len(raw) - 1
        assert gone["date"] not in set(df.loc[df["symbol"] == "BBB", "date"])
        assert df["close"].notna().all()

    def test_missing_volume_is_zero_and_prices_are_never_filled_from_other_days(self, provider_from_frame):
        raw = raw_frame()
        row = raw[raw["symbol"] == "AAA"].iloc[5]
        raw.loc[row.name, "volume"] = np.nan
        df = provider_from_frame(raw).get_historical_data()
        filled = df[(df["symbol"] == "AAA") & (df["date"] == row["date"])].iloc[0]
        assert filled["volume"] == 0.0
        assert filled["close"] == row["close"]  # the price itself is untouched

    # ------------------------------------------------------------------ broken data is refused, never repaired
    @pytest.mark.parametrize(
        "column, value, label",
        [
            ("high", 1.0, "high below low"),
            ("close", 0.0, "zero close"),
            ("close", -5.0, "negative close"),
            ("open", 0.0, "zero open"),
            ("open", -1.0, "negative open"),
            ("volume", -10.0, "negative volume"),
        ],
    )
    def test_impossible_data_is_rejected(self, provider_from_frame, column, value, label):
        raw = raw_frame()
        raw.loc[7, column] = value
        with pytest.raises(DataValidationError):
            provider_from_frame(raw).get_historical_data()

    def test_a_missing_required_column_is_rejected(self, provider_from_frame):
        with pytest.raises(DataValidationError, match="Missing required column"):
            provider_from_frame(raw_frame().drop(columns=["volume"])).get_historical_data()

    def test_an_unparseable_date_is_rejected(self, provider_from_frame):
        raw = raw_frame().astype({"date": object})
        raw.loc[2, "date"] = "not-a-date"
        with pytest.raises(DataValidationError, match="date"):
            provider_from_frame(raw).get_historical_data()


# ======================================================================================================================
# The local provider
# ======================================================================================================================
class TestLocalProviderContract(ProviderContract):
    @pytest.fixture
    def provider(self):
        return LocalDataProvider(DEV_PRICES)

    @pytest.fixture
    def provider_from_frame(self, tmp_path):
        def build(frame: pd.DataFrame) -> LocalDataProvider:
            path = tmp_path / "prices.csv"
            frame.to_csv(path, index=False)
            return LocalDataProvider(path)

        return build


class TestLocalProviderSpecifics:
    """Behaviour that belongs to reading a file, not to every provider."""

    def test_the_development_file_is_recognised_as_synthetic(self):
        provider = LocalDataProvider(DEV_PRICES)
        assert provider.name == "local" and provider.is_synthetic is True

    def test_a_file_without_synthetic_in_its_name_is_not_labelled_synthetic(self, tmp_path):
        path = tmp_path / "prices.csv"
        raw_frame().to_csv(path, index=False)
        assert LocalDataProvider(path).is_synthetic is False

    def test_parquet_files_are_read_too(self, tmp_path):
        pytest.importorskip("pyarrow")
        path = tmp_path / "prices.parquet"
        raw_frame().to_parquet(path, index=False)
        assert list(LocalDataProvider(path).get_historical_data().columns) == REQUIRED_COLUMNS

    def test_a_missing_file_fails_the_availability_check_and_the_load(self, tmp_path):
        provider = LocalDataProvider(tmp_path / "nope.csv")
        with pytest.raises(DataUnavailableError, match="not found"):
            provider.check_available()
        with pytest.raises(FileNotFoundError):
            provider.get_historical_data()

    def test_the_availability_message_does_not_contain_the_path(self, tmp_path):
        with pytest.raises(DataUnavailableError) as caught:
            LocalDataProvider(tmp_path / "secret_folder" / "nope.csv").check_available()
        assert "secret_folder" not in str(caught.value) and str(tmp_path) not in str(caught.value)

    def test_duplicates_are_resolved_by_keeping_the_last_row(self, tmp_path):
        raw = raw_frame()
        duplicate = raw.iloc[[3]].copy()
        duplicate["close"] = raw.loc[3, "close"]
        duplicate["volume"] = 99999.0
        path = tmp_path / "prices.csv"
        pd.concat([raw, duplicate]).to_csv(path, index=False)
        df = LocalDataProvider(path).get_historical_data()
        assert len(df) == len(raw)
        assert df[(df["symbol"] == raw.loc[3, "symbol"]) & (df["date"] == raw.loc[3, "date"])]["volume"].iloc[0] == 99999.0
