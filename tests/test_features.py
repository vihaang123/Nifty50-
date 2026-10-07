"""
Tests for Phase 2 (src/features.py).

Every feature is checked against a number worked out by hand (or by a separate
plain-numpy calculation) on a small deterministic dataset. A test that only
checked "the function runs" would pass even if the maths were wrong.

Run:  pytest -q
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import (
    FEATURE_COLUMNS,
    FIRST_VALID_ROW,
    MIN_HISTORY_ROWS,
    OUTPUT_COLUMNS,
    TRADING_DAYS_PER_YEAR,
    build_features,
    clean_feature_data,
    missing_value_report,
    print_demo,
)
from src.sample_data import generate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQRT_252 = np.sqrt(TRADING_DAYS_PER_YEAR)


# ======================= helpers: tiny deterministic datasets =======================
def make_stock(closes, symbol="AAA", volumes=None, start="2024-01-01"):
    """A valid Phase 1 style table for one stock, built from a list of closes."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    volumes = np.full(n, 1000.0) if volumes is None else np.asarray(volumes, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=n),
            "symbol": symbol,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volumes,
        }
    )


def make_market(closes, start="2024-01-01"):
    return make_stock(closes, symbol="MARKET_INDEX", volumes=np.zeros(len(closes)), start=start)


def closes_from_returns(returns, start=100.0):
    """Closes whose daily returns are exactly `returns` (closes has one more entry than returns)."""
    return start * np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(returns, dtype=float))])


def random_closes(n, seed, daily_vol=0.01):
    rng = np.random.default_rng(seed)
    return closes_from_returns(daily_vol * rng.standard_normal(n - 1))


def random_volumes(n, seed):
    return np.random.default_rng(seed).uniform(500, 1500, n)


def one_symbol(features, symbol="AAA"):
    return features[features["symbol"] == symbol].reset_index(drop=True)


def daily_returns(closes):
    closes = np.asarray(closes, dtype=float)
    return closes[1:] / closes[:-1] - 1.0  # entry i is the return on row i + 1


@pytest.fixture
def market_200():
    return make_market(random_closes(200, seed=99))


@pytest.fixture
def stock_200():
    return make_stock(random_closes(200, seed=1), volumes=random_volumes(200, seed=2))


# ======================= output structure =======================
def test_output_has_the_expected_columns_in_order(stock_200, market_200):
    features = build_features(stock_200, market_200)
    assert list(features.columns) == OUTPUT_COLUMNS
    assert OUTPUT_COLUMNS[:3] == ["date", "symbol", "close"]
    assert len(FEATURE_COLUMNS) == 13


def test_rows_are_kept_and_close_is_unchanged(stock_200, market_200):
    features = build_features(stock_200, market_200)
    assert len(features) == len(stock_200)
    np.testing.assert_array_equal(features["close"].to_numpy(), stock_200["close"].to_numpy())
    np.testing.assert_array_equal(features["date"].to_numpy(), stock_200["date"].to_numpy())


def test_input_tables_are_not_modified(stock_200, market_200):
    stock_before, market_before = stock_200.copy(), market_200.copy()
    build_features(stock_200, market_200)
    pd.testing.assert_frame_equal(stock_200, stock_before)
    pd.testing.assert_frame_equal(market_200, market_before)


def test_row_order_of_the_inputs_does_not_matter(market_200):
    """Shuffled stock rows and shuffled market rows give the same result: matching is by date."""
    two = pd.concat(
        [make_stock(random_closes(200, 1), "AAA", random_volumes(200, 2)), make_stock(random_closes(200, 3), "BBB", random_volumes(200, 4))],
        ignore_index=True,
    )
    expected = build_features(two, market_200)
    shuffled = build_features(two.sample(frac=1, random_state=0), market_200.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(shuffled, expected)
    assert expected.sort_values(["symbol", "date"]).index.equals(expected.index)  # sorted by symbol, date


# ======================= 1. returns =======================
def test_returns_on_constant_growth():
    """Price grows 1% every day, so an N-day return is exactly 1.01**N - 1 on every valid row."""
    n = 90
    features = build_features(make_stock(100 * 1.01 ** np.arange(n)), make_market(random_closes(n, 5)))
    for column, days in [("return_5d", 5), ("return_20d", 20), ("return_60d", 60)]:
        valid = features[column].iloc[days:]
        assert valid.to_numpy() == pytest.approx(1.01**days - 1, rel=1e-9)
        assert features[column].iloc[:days].isna().all()  # not enough history yet


def test_returns_on_a_simple_ladder_by_hand():
    """Closes 100, 101, 102, ... so on row 10 the close is 110 and 5 rows earlier it was 105."""
    n = 90
    features = build_features(make_stock(100 + np.arange(n)), make_market(random_closes(n, 5)))
    assert features["return_5d"].iloc[10] == pytest.approx(110 / 105 - 1)
    assert features["return_20d"].iloc[30] == pytest.approx(130 / 110 - 1)
    assert features["return_60d"].iloc[80] == pytest.approx(180 / 120 - 1)


def test_return_can_be_negative():
    closes = np.full(70, 100.0)
    closes[10:] = 90.0  # a 10% drop on row 10
    features = build_features(make_stock(closes), make_market(random_closes(70, 5)))
    assert features["return_5d"].iloc[10] == pytest.approx(-0.10)
    assert features["return_5d"].iloc[15] == pytest.approx(0.0)  # 5 rows later both ends are at 90


# ======================= 2. volatility =======================
def test_volatility_zero_when_returns_are_constant():
    n = 90
    features = build_features(make_stock(100 * 1.01 ** np.arange(n)), make_market(random_closes(n, 5)))
    assert features["volatility_20d"].iloc[20:].to_numpy() == pytest.approx(0.0, abs=1e-9)
    assert features["volatility_60d"].iloc[60:].to_numpy() == pytest.approx(0.0, abs=1e-9)


def test_volatility_by_hand_with_alternating_up_and_down_days():
    """
    Returns alternate +2%, -2%. In any 20-day window that is 10 of each, so the mean is 0 and the
    sample standard deviation is 0.02 * sqrt(20/19). Annualised: times sqrt(252).
    For 60 days: 0.02 * sqrt(60/59).
    """
    n = 100
    closes = [100.0]
    for day in range(1, n):
        closes.append(closes[-1] * (1.02 if day % 2 == 1 else 0.98))
    features = build_features(make_stock(closes), make_market(random_closes(n, 5)))
    expected_20 = 0.02 * np.sqrt(20 / 19) * SQRT_252
    expected_60 = 0.02 * np.sqrt(60 / 59) * SQRT_252
    assert features["volatility_20d"].iloc[20:].to_numpy() == pytest.approx(expected_20, rel=1e-9)
    assert features["volatility_60d"].iloc[60:].to_numpy() == pytest.approx(expected_60, rel=1e-9)


def test_volatility_matches_plain_numpy_on_random_prices():
    n = 150
    closes = random_closes(n, seed=7, daily_vol=0.02)
    features = build_features(make_stock(closes), make_market(random_closes(n, 5)))
    r = daily_returns(closes)
    for row in (20, 60, 100, 149):
        window20 = r[row - 20 : row]  # the 20 returns ending on `row`
        assert features["volatility_20d"].iloc[row] == pytest.approx(np.std(window20, ddof=1) * SQRT_252, rel=1e-9)
    for row in (60, 100, 149):
        window60 = r[row - 60 : row]
        assert features["volatility_60d"].iloc[row] == pytest.approx(np.std(window60, ddof=1) * SQRT_252, rel=1e-9)


# ======================= 3a. moving-average ratios =======================
def test_ma_ratios_by_hand_on_a_ladder():
    """
    Closes 100 + t. The mean of the last 20 closes ending on row t is 100 + t - 9.5,
    and of the last 50 it is 100 + t - 24.5.
    """
    n = 90
    features = build_features(make_stock(100 + np.arange(n)), make_market(random_closes(n, 5)))
    assert features["price_ma20_ratio"].iloc[19] == pytest.approx(119 / 109.5)  # first valid row: mean of 100..119
    assert features["price_ma20_ratio"].iloc[60] == pytest.approx(160 / 150.5)
    assert features["price_ma50_ratio"].iloc[49] == pytest.approx(149 / 124.5)  # mean of 100..149
    assert features["price_ma50_ratio"].iloc[80] == pytest.approx(180 / 155.5)
    assert features["price_ma20_ratio"].iloc[:19].isna().all()
    assert features["price_ma50_ratio"].iloc[:49].isna().all()


def test_ma_ratio_is_one_for_a_flat_price_and_above_one_after_a_jump():
    n = 80
    flat = build_features(make_stock(np.full(n, 50.0)), make_market(random_closes(n, 5)))
    assert flat["price_ma20_ratio"].iloc[19:].to_numpy() == pytest.approx(1.0)
    assert flat["price_ma50_ratio"].iloc[49:].to_numpy() == pytest.approx(1.0)

    closes = np.full(n, 100.0)
    closes[60:] = 120.0  # jump on row 60
    jump = build_features(make_stock(closes), make_market(random_closes(n, 5)))
    # row 60: last 20 closes are 19 x 100 and 1 x 120 -> mean 101
    assert jump["price_ma20_ratio"].iloc[60] == pytest.approx(120 / 101)
    assert jump["price_ma20_ratio"].iloc[60] > 1


# ======================= 3b. RSI =======================
def rsi_series(closes):
    n = len(closes)
    return build_features(make_stock(closes), make_market(random_closes(n, 5)))["rsi_14"]


def test_rsi_by_hand():
    """
    14 changes: ten days of +2 and four days of -3.
    avg_gain = 20/14, avg_loss = 12/14, RS = 20/12 = 5/3, RSI = 100 - 100/(1 + 5/3) = 62.5.
    """
    changes = [2.0] * 10 + [-3.0] * 4
    closes = 100 + np.concatenate([[0.0], np.cumsum(changes)])
    closes = np.concatenate([closes, np.full(70 - len(closes), closes[-1])])  # pad so the dataset is long enough
    rsi = rsi_series(closes)
    assert rsi.iloc[14] == pytest.approx(62.5)
    assert rsi.iloc[:14].isna().all()  # 14 changes need 15 prices


def test_rsi_extremes_and_flat():
    n = 40
    assert rsi_series(100 + np.arange(n)).iloc[14:].to_numpy() == pytest.approx(100.0)  # only gains -> 100
    assert rsi_series(200 - np.arange(n)).iloc[14:].to_numpy() == pytest.approx(0.0)  # only losses -> 0
    assert rsi_series(np.full(n, 100.0)).iloc[14:].to_numpy() == pytest.approx(50.0)  # flat: 0 gain, 0 loss -> 50, not a crash


def test_rsi_is_50_when_gains_equal_losses():
    closes = 100 + np.array([0, 1] * 20, dtype=float)  # +1, -1, +1, -1 ...
    assert rsi_series(closes).iloc[14:].to_numpy() == pytest.approx(50.0)


def test_rsi_stays_between_0_and_100_on_random_prices():
    rsi = rsi_series(random_closes(500, seed=11, daily_vol=0.03)).dropna()
    assert len(rsi) == 500 - 14
    assert rsi.between(0, 100).all()


# ======================= 4. drawdown =======================
CRASH_PATH = np.array([100, 110, 120, 105, 90] + [110] * 125, dtype=float)  # peak 120 on row 2, trough 90 on row 4


def drawdown_series(closes):
    n = len(closes)
    return build_features(make_stock(closes), make_market(random_closes(n, 5)))["max_drawdown"]


def test_drawdown_by_hand():
    """Peak 120 then trough 90: the fall is 90/120 - 1 = -25%. Later days at 110 are a smaller fall."""
    dd = drawdown_series(CRASH_PATH)
    assert dd.iloc[:59].isna().all()  # the 60-day window is not full until row 59
    assert dd.iloc[59] == pytest.approx(-0.25)
    assert dd.iloc[60] == pytest.approx(-0.25)
    assert dd.iloc[61] == pytest.approx(-0.25)


def test_drawdown_forgets_old_peaks_as_the_window_moves():
    """
    The window is the last 60 rows. Row 62 covers rows 3..62: peak 105, trough 90 -> 90/105 - 1.
    Row 63 covers rows 4..63: it starts at the trough, so no fall is visible -> 0.
    """
    dd = drawdown_series(CRASH_PATH)
    assert dd.iloc[62] == pytest.approx(90 / 105 - 1)
    assert dd.iloc[63] == pytest.approx(0.0)
    assert dd.iloc[64:].to_numpy() == pytest.approx(0.0)


def test_drawdown_is_zero_for_a_rising_price_and_never_positive_otherwise():
    assert drawdown_series(100 + np.arange(100)).iloc[59:].to_numpy() == pytest.approx(0.0)
    random_dd = drawdown_series(random_closes(400, seed=21, daily_vol=0.03)).dropna()
    assert (random_dd <= 0).all() and (random_dd > -1).all()


def test_drawdown_ignores_a_later_peak():
    """A price that crashes and then rockets up: the crash is still the worst fall inside the window."""
    closes = np.array([100.0, 80.0] + [200.0] * 58)  # 60 rows: fell 20%, then more than doubled
    dd = drawdown_series(np.concatenate([closes, np.full(20, 200.0)]))
    assert dd.iloc[59] == pytest.approx(80 / 100 - 1)


# ======================= 5/6. beta and market correlation =======================
def test_beta_and_correlation_when_stock_moves_exactly_twice_the_market():
    n = 150
    m = 0.01 * np.random.default_rng(3).standard_normal(n - 1)
    features = build_features(make_stock(closes_from_returns(2 * m)), make_market(closes_from_returns(m)))
    assert features["beta_60d"].iloc[60:].to_numpy() == pytest.approx(2.0, rel=1e-8)
    assert features["market_correlation_60d"].iloc[60:].to_numpy() == pytest.approx(1.0, rel=1e-8)
    assert features["beta_60d"].iloc[:60].isna().all()
    assert features["market_correlation_60d"].iloc[:60].isna().all()


def test_beta_and_correlation_when_stock_moves_against_the_market():
    n = 150
    m = 0.01 * np.random.default_rng(4).standard_normal(n - 1)
    features = build_features(make_stock(closes_from_returns(-0.5 * m)), make_market(closes_from_returns(m)))
    assert features["beta_60d"].iloc[60:].to_numpy() == pytest.approx(-0.5, rel=1e-8)
    assert features["market_correlation_60d"].iloc[60:].to_numpy() == pytest.approx(-1.0, rel=1e-8)


def test_beta_and_correlation_match_plain_numpy_on_random_data():
    n = 200
    stock_closes, market_closes = random_closes(n, seed=5, daily_vol=0.02), random_closes(n, seed=6)
    features = build_features(make_stock(stock_closes), make_market(market_closes))
    s, m = daily_returns(stock_closes), daily_returns(market_closes)
    for row in (60, 100, 199):
        sw, mw = s[row - 60 : row], m[row - 60 : row]  # the 60 returns ending on `row`
        assert features["beta_60d"].iloc[row] == pytest.approx(np.cov(sw, mw)[0, 1] / np.var(mw, ddof=1), rel=1e-9)
        assert features["market_correlation_60d"].iloc[row] == pytest.approx(np.corrcoef(sw, mw)[0, 1], rel=1e-9)


def test_market_is_aligned_by_date_even_when_the_stock_is_missing_days():
    """
    The stock is always 0.01 x the market, so over ANY interval its return equals the market's.
    Remove three days from the stock. The stock's return across each gap must then be compared with
    the market's return across the same gap, which gives beta 1 and correlation 1 exactly.
    (Comparing it with the market's one-day return instead would break this.)
    """
    n = 200
    market_closes = random_closes(n, seed=8)
    market = make_market(market_closes)
    stock = make_stock(market_closes * 0.01).drop(index=[70, 90, 120]).reset_index(drop=True)
    features = build_features(stock, market)
    valid = features.dropna(subset=["beta_60d"])
    assert len(valid) > 100
    assert valid["beta_60d"].to_numpy() == pytest.approx(1.0, rel=1e-8)
    assert valid["market_correlation_60d"].to_numpy() == pytest.approx(1.0, rel=1e-8)


def test_market_dates_the_stock_does_not_have_are_ignored():
    n = 150
    market_closes = random_closes(n, seed=9)
    stock = make_stock(random_closes(n, seed=10))
    # same dates, but give the market 30 extra days after the stock's last day
    market_longer = make_market(np.concatenate([market_closes, random_closes(30, seed=12)]))
    pd.testing.assert_frame_equal(build_features(stock, make_market(market_closes)), build_features(stock, market_longer))


def test_missing_market_value_gives_missing_beta_not_a_guess():
    n = 200
    market = make_market(random_closes(n, seed=13)).drop(index=100).reset_index(drop=True)  # market has no row for stock day 100
    features = build_features(make_stock(random_closes(n, seed=14)), market)
    assert not np.isnan(features["beta_60d"].iloc[99])  # window ends before the gap
    assert features["beta_60d"].iloc[100:160].isna().all()  # any window touching the gap is incomplete
    assert not np.isnan(features["beta_60d"].iloc[199])  # recovers once the gap leaves the window


def test_flat_market_gives_missing_beta_not_infinity():
    n = 100
    features = build_features(make_stock(random_closes(n, seed=15)), make_market(np.full(n, 10_000.0)))
    assert features["beta_60d"].isna().all()  # variance of the market is 0, so beta is undefined
    assert not np.isinf(features[FEATURE_COLUMNS].to_numpy(dtype=float)).any()


# ======================= 7. volume =======================
def volume_features(volumes):
    n = len(volumes)
    return build_features(make_stock(random_closes(n, 5), volumes=volumes), make_market(random_closes(n, 6)))


def test_constant_volume_gives_zero_change():
    features = volume_features(np.full(60, 1000.0))
    assert features["avg_volume_20d"].iloc[19:].to_numpy() == pytest.approx(1000.0)
    assert features["volume_change"].iloc[19:].to_numpy() == pytest.approx(0.0)
    assert features["avg_volume_20d"].iloc[:19].isna().all()


def test_volume_change_by_hand_on_a_spike():
    """19 days at 100 then 300: average = (19*100 + 300)/20 = 110, change = 300/110 - 1."""
    volumes = np.array([100.0] * 19 + [300.0] + [100.0] * 20)
    features = volume_features(volumes)
    assert features["avg_volume_20d"].iloc[19] == pytest.approx(110.0)
    assert features["volume_change"].iloc[19] == pytest.approx(300 / 110 - 1)
    # row 20 drops the first 100 and adds another 100: the spike is still in the window
    assert features["avg_volume_20d"].iloc[20] == pytest.approx(110.0)
    assert features["volume_change"].iloc[20] == pytest.approx(100 / 110 - 1)


def test_zero_volume_never_causes_division_errors():
    all_zero = volume_features(np.zeros(60))
    assert (all_zero["avg_volume_20d"].iloc[19:] == 0).all()
    assert all_zero["volume_change"].isna().all()  # 0 / 0 -> missing, not infinity or an error
    assert not np.isinf(all_zero[FEATURE_COLUMNS].to_numpy(dtype=float)).any()

    # some zero days inside a window: 10 days of 100 and 10 days of 0 -> average 50; today is 0 -> change -100%
    mixed = volume_features(np.array([100.0] * 10 + [0.0] * 10 + [100.0] * 20))
    assert mixed["avg_volume_20d"].iloc[19] == pytest.approx(50.0)
    assert mixed["volume_change"].iloc[19] == pytest.approx(-1.0)


# ======================= missing values =======================
@pytest.fixture
def two_stocks_features(market_200):
    two = pd.concat(
        [
            make_stock(random_closes(200, seed=1), "AAA", random_volumes(200, 2)),
            make_stock(random_closes(150, seed=3), "BBB", random_volumes(150, 4)),
        ],
        ignore_index=True,
    )
    return build_features(two, market_200)


def test_each_feature_becomes_valid_on_exactly_the_documented_row(two_stocks_features):
    for symbol in ("AAA", "BBB"):  # each stock starts counting from ITS OWN first row
        one = one_symbol(two_stocks_features, symbol)
        for column, first_valid in FIRST_VALID_ROW.items():
            assert one[column].iloc[:first_valid].isna().all(), f"{symbol} {column}: should be missing before row {first_valid}"
            assert one[column].iloc[first_valid:].notna().all(), f"{symbol} {column}: should be present from row {first_valid}"


def test_warmup_is_left_missing_and_never_filled(two_stocks_features):
    bbb = one_symbol(two_stocks_features, "BBB")  # starts right after AAA in the table
    assert bbb.loc[:59, FEATURE_COLUMNS].isna().any(axis=1).all()  # still incomplete, no history borrowed from AAA
    assert bbb["return_60d"].iloc[:60].isna().all()


def test_missing_value_report_counts(two_stocks_features):
    table, rows_with_any_missing = missing_value_report(two_stocks_features)
    assert list(table.index) == FEATURE_COLUMNS
    for column, first_valid in FIRST_VALID_ROW.items():
        assert table.loc[column, "missing_rows"] == 2 * first_valid  # two stocks
    assert rows_with_any_missing == 2 * MIN_HISTORY_ROWS
    assert table.loc["return_60d", "missing_%"] == pytest.approx(120 / 350 * 100, abs=0.01)


def test_clean_removes_exactly_the_first_60_rows_of_each_stock(two_stocks_features):
    assert MIN_HISTORY_ROWS == 60
    cleaned = clean_feature_data(two_stocks_features)
    assert len(cleaned) == (200 - 60) + (150 - 60)
    assert cleaned[FEATURE_COLUMNS].notna().all().all()
    assert not np.isinf(cleaned[FEATURE_COLUMNS].to_numpy(dtype=float)).any()
    assert list(cleaned.index) == list(range(len(cleaned)))
    first_rows = cleaned.groupby("symbol")["date"].min()
    all_dates = pd.bdate_range("2024-01-01", periods=200)
    assert first_rows["AAA"] == all_dates[60] and first_rows["BBB"] == all_dates[60]


def test_clean_keeps_the_remaining_values_unchanged(two_stocks_features):
    cleaned = clean_feature_data(two_stocks_features)
    original = two_stocks_features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    pd.testing.assert_frame_equal(cleaned, original)


def test_a_stock_with_too_little_history_disappears_after_cleaning(market_200):
    short = make_stock(random_closes(60, seed=3), "SHORT", random_volumes(60, 4))  # 60 rows < 61 needed
    ok = make_stock(random_closes(120, seed=1), "OK", random_volumes(120, 2))
    cleaned = clean_feature_data(build_features(pd.concat([short, ok], ignore_index=True), market_200))
    assert set(cleaned["symbol"]) == {"OK"}


# ======================= no future leakage =======================
@pytest.fixture
def two_stocks_200():
    return pd.concat(
        [
            make_stock(random_closes(200, seed=1), "AAA", random_volumes(200, 2)),
            make_stock(random_closes(200, seed=3), "BBB", random_volumes(200, 4)),
        ],
        ignore_index=True,
    )


CUT = 150  # rows from this index onward are the "future" in the tests below


def assert_features_before_cut_unchanged(original, changed, cut_date):
    before_a = original[original["date"] < cut_date].reset_index(drop=True)
    before_b = changed[changed["date"] < cut_date].reset_index(drop=True)
    assert len(before_a) > 0
    pd.testing.assert_frame_equal(before_a, before_b, check_exact=False, rtol=1e-12, atol=0)


def test_changing_future_stock_prices_does_not_change_earlier_features(two_stocks_200, market_200):
    cut_date = pd.bdate_range("2024-01-01", periods=200)[CUT]
    baseline = build_features(two_stocks_200, market_200)

    changed = two_stocks_200.copy()
    future = changed["date"] >= cut_date
    changed.loc[future, "close"] = changed.loc[future, "close"] * np.random.default_rng(0).uniform(0.3, 3.0, future.sum())
    after = build_features(changed, market_200)

    assert_features_before_cut_unchanged(baseline, after, cut_date)
    # the test is sensitive: the change must be visible from the cut date on
    day_of_change = lambda f: f[(f["symbol"] == "AAA") & (f["date"] == cut_date)]["return_5d"].iloc[0]
    assert day_of_change(baseline) != day_of_change(after)


def test_changing_future_volume_does_not_change_earlier_features(two_stocks_200, market_200):
    cut_date = pd.bdate_range("2024-01-01", periods=200)[CUT]
    baseline = build_features(two_stocks_200, market_200)
    changed = two_stocks_200.copy()
    changed.loc[changed["date"] >= cut_date, "volume"] = 10_000_000.0
    after = build_features(changed, market_200)
    assert_features_before_cut_unchanged(baseline, after, cut_date)
    assert baseline["avg_volume_20d"].iloc[CUT] != after["avg_volume_20d"].iloc[CUT]


def test_changing_future_market_prices_does_not_change_earlier_features(two_stocks_200, market_200):
    cut_date = pd.bdate_range("2024-01-01", periods=200)[CUT]
    baseline = build_features(two_stocks_200, market_200)
    changed_market = market_200.copy()
    future = changed_market["date"] >= cut_date
    changed_market.loc[future, "close"] = changed_market.loc[future, "close"] * np.random.default_rng(1).uniform(0.5, 2.0, future.sum())
    after = build_features(two_stocks_200, changed_market)
    assert_features_before_cut_unchanged(baseline, after, cut_date)
    assert not np.allclose(baseline["beta_60d"].iloc[CUT:].dropna(), after["beta_60d"].iloc[CUT:].dropna())


def test_cutting_the_data_off_gives_the_same_features_as_the_full_run(two_stocks_200, market_200):
    """
    The strongest leakage check. If the feature on date t used anything after t, running on data that
    stops at t would give a different answer from running on all the data. It must not.
    """
    cut_date = pd.bdate_range("2024-01-01", periods=200)[120]
    full = build_features(two_stocks_200, market_200)
    truncated = build_features(two_stocks_200[two_stocks_200["date"] <= cut_date], market_200[market_200["date"] <= cut_date])
    full_up_to_cut = full[full["date"] <= cut_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(truncated, full_up_to_cut, check_exact=False, rtol=1e-12, atol=0)


def test_cutting_off_is_also_safe_on_the_synthetic_dev_generator():
    prices, index = generate(seed=3, start="2020-01-01", end="2021-06-30")
    cut_date = pd.Timestamp("2021-01-15")
    full = build_features(prices, index)
    truncated = build_features(prices[prices["date"] <= cut_date], index[index["date"] <= cut_date])
    expected = full[full["date"] <= cut_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(truncated, expected, check_exact=False, rtol=1e-12, atol=0)


# ======================= input checks =======================
def test_bad_inputs_are_rejected_with_clear_messages(stock_200, market_200):
    with pytest.raises(ValueError, match="volume"):
        build_features(stock_200.drop(columns=["volume"]), market_200)
    with pytest.raises(ValueError, match="duplicate"):
        build_features(pd.concat([stock_200, stock_200.iloc[[0]]]), market_200)
    with pytest.raises(ValueError, match="close"):
        build_features(stock_200.assign(close=np.where(stock_200.index == 5, np.nan, stock_200["close"])), market_200)
    with pytest.raises(ValueError, match="close"):
        build_features(stock_200, market_200.drop(columns=["close"]))
    with pytest.raises(ValueError, match="single market index"):
        build_features(stock_200, pd.concat([market_200, market_200.assign(symbol="OTHER_INDEX")]))
    with pytest.raises(ValueError, match="duplicate dates"):
        build_features(stock_200, pd.concat([market_200, market_200.iloc[[0]]]))


# ======================= the development dataset end to end =======================
@pytest.fixture(scope="module")
def dev_features():
    config = load_config(PROJECT_ROOT / "config.yaml")
    stocks = get_provider(config, PROJECT_ROOT).get_historical_data()
    market = LocalDataProvider(PROJECT_ROOT / config["data"]["market_index_file"]).get_historical_data()
    return stocks, build_features(stocks, market)


def test_dev_data_features_are_complete_and_finite_after_cleaning(dev_features):
    stocks, features = dev_features
    cleaned = clean_feature_data(features)
    n_stocks = stocks["symbol"].nunique()
    assert len(features) == len(stocks)
    assert len(features) - len(cleaned) == MIN_HISTORY_ROWS * n_stocks
    assert cleaned[FEATURE_COLUMNS].notna().all().all()
    assert np.isfinite(cleaned[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
    assert cleaned["symbol"].nunique() == n_stocks


def test_dev_data_features_stay_inside_their_possible_ranges(dev_features):
    cleaned = clean_feature_data(dev_features[1])
    assert cleaned["rsi_14"].between(0, 100).all()
    assert cleaned["market_correlation_60d"].between(-1 - 1e-9, 1 + 1e-9).all()
    assert (cleaned["max_drawdown"] <= 0).all() and (cleaned["max_drawdown"] > -1).all()
    assert (cleaned[["volatility_20d", "volatility_60d", "avg_volume_20d"]] >= 0).all().all()
    assert (cleaned[["price_ma20_ratio", "price_ma50_ratio"]] > 0).all().all()
    assert (cleaned[["return_5d", "return_20d", "return_60d"]] > -1).all().all()
    assert (cleaned["volume_change"] > -1 - 1e-9).all()


# ======================= demonstration output =======================
def test_demo_prints_the_report_and_returns_the_cleaned_table(capsys):
    prices, index = generate(seed=1, start="2020-01-01", end="2020-12-31")
    cleaned = print_demo(prices, index, source="dev_prices_synthetic.csv")
    out = capsys.readouterr().out
    for expected in [
        "=== FEATURE ENGINEERING ===",
        "SYNTHETIC".lower(),
        "Input rows:",
        "Output rows:",
        "Features created:",
        "- return_5d",
        "- volume_change",
        "Missing values before cleaning",
        "Rows removed due to insufficient history:",
        "Final feature rows:",
        "Feature summary:",
    ]:
        assert expected in out
    assert len(cleaned) == len(prices) - MIN_HISTORY_ROWS * prices["symbol"].nunique()
