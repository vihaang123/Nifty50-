"""
Tests for Phase 3 (src/pca_model.py).

Most tests use a small made-up feature table whose columns have very different scales
(RSI around 50, returns around 0.02, volume in the millions). Some use the project's
synthetic development data end to end.

Run:  pytest -q
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.decomposition import PCA

from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import FEATURE_COLUMNS, build_features, clean_feature_data
from src.pca_model import (
    PCA_FEATURES,
    VOLUME_FEATURE,
    fit_pca,
    get_explained_variance,
    get_feature_loadings,
    interpretation_text,
    main,
    plot_explained_variance,
    plot_pc_scatter,
    preprocess,
    prepare_features,
    print_pca_report,
    sample_for_plot,
    save_pca_output,
    top_loadings,
    transform_pca,
)
from src.sample_data import DEV_STOCKS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_FEATURES = 13


# ======================= helpers =======================
def make_feature_df(per_symbol=100, seed=0, symbols=("AAA", "BBB", "CCC", "DDD")):
    """
    A made-up Phase 2 style table (date, symbol, close + 13 features) with realistic *scales*:
    returns ~0.02, volatility ~0.25, RSI ~50, beta ~1, volume ~ millions and skewed.
    Two hidden factors (z1 = 'trend', z2 = 'risk') make some columns move together.
    """
    rng = np.random.default_rng(seed)
    n = per_symbol * len(symbols)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)

    def noise(scale):
        return scale * rng.standard_normal(n)

    df = pd.DataFrame(
        {
            "date": np.tile(pd.bdate_range("2022-01-03", periods=per_symbol).to_numpy(), len(symbols)),
            "symbol": np.repeat(symbols, per_symbol),
            "close": rng.uniform(50, 150, n),
            "return_5d": 0.02 * z1 + noise(0.01),
            "return_20d": 0.05 * z1 + noise(0.02),
            "return_60d": 0.10 * z1 + noise(0.04),
            "volatility_20d": 0.25 + 0.05 * z2 + noise(0.02),
            "volatility_60d": 0.25 + 0.04 * z2 + noise(0.01),
            "price_ma20_ratio": 1 + 0.03 * z1 + noise(0.01),
            "price_ma50_ratio": 1 + 0.05 * z1 + noise(0.02),
            "rsi_14": np.clip(50 + 15 * z1 + noise(5), 1, 99),
            "max_drawdown": -(0.10 + 0.04 * np.abs(z2)),
            "beta_60d": 1 + 0.3 * z2 + noise(0.1),
            "market_correlation_60d": np.clip(0.6 + 0.1 * z2 + noise(0.05), -0.99, 0.99),
            "avg_volume_20d": np.exp(14 + rng.standard_normal(n)),  # about 1.2 million, strongly right-skewed
            "volume_change": noise(0.3),
        }
    )
    assert set(PCA_FEATURES) <= set(df.columns)
    return df


@pytest.fixture
def feature_df():
    return make_feature_df()


def pc_columns(df):
    return [c for c in df.columns if re.fullmatch(r"PC\d+", c)]


# ======================= the 13 input features =======================
def test_pca_uses_exactly_the_13_phase2_features():
    assert PCA_FEATURES == list(FEATURE_COLUMNS)
    assert len(PCA_FEATURES) == N_FEATURES
    assert VOLUME_FEATURE == "avg_volume_20d"


# ======================= test 1: number of components =======================
@pytest.mark.parametrize("n_components", [1, 3, 5, 10, 13])
def test_model_has_the_requested_number_of_components(feature_df, n_components):
    model = fit_pca(feature_df, n_components=n_components)
    assert model.n_components == n_components
    assert model.pca.components_.shape == (n_components, N_FEATURES)
    assert model.component_names == [f"PC{i}" for i in range(1, n_components + 1)]
    assert len(get_explained_variance(model)) == n_components
    assert pc_columns(transform_pca(feature_df, model)) == model.component_names


def test_default_is_five_components(feature_df):
    assert fit_pca(feature_df).n_components == 5


@pytest.mark.parametrize("bad", [0, 14, -1, 2.5, True, "5", None])
def test_invalid_number_of_components_is_rejected(feature_df, bad):
    with pytest.raises(ValueError, match="n_components"):
        fit_pca(feature_df, n_components=bad)


def test_too_few_rows_for_the_components_is_rejected(feature_df):
    with pytest.raises(ValueError, match="rows"):
        fit_pca(feature_df.head(5), n_components=5)


# ======================= test 2: explained variance =======================
def test_explained_variance_ratios_are_valid(feature_df):
    model = fit_pca(feature_df, n_components=5)
    table = get_explained_variance(model)
    ratio, cumulative = table["explained_variance_ratio"], table["cumulative_explained_variance"]
    assert list(table.columns) == ["explained_variance_ratio", "cumulative_explained_variance"]
    assert ((ratio > 0) & (ratio <= 1)).all()
    assert ratio.is_monotonic_decreasing  # PC1 explains the most, then PC2, ...
    assert cumulative.iloc[-1] == pytest.approx(ratio.sum())
    assert cumulative.is_monotonic_increasing
    assert 0 < cumulative.iloc[-1] < 1  # 5 of 13 components cannot keep everything on this data


def test_all_13_components_explain_exactly_everything(feature_df):
    ratio = get_explained_variance(fit_pca(feature_df, n_components=13))["explained_variance_ratio"]
    assert ratio.sum() == pytest.approx(1.0, abs=1e-12)


def test_explained_variance_matches_an_independent_eigenvalue_calculation(feature_df):
    """
    PCA on standardised data = eigen-decomposition of the correlation matrix.
    Component k's share is eigenvalue_k / sum of eigenvalues (= 13, one per standardised feature).
    """
    model = fit_pca(feature_df, n_components=13)
    prepared = prepare_features(feature_df)
    eigenvalues = np.sort(np.linalg.eigvalsh(np.corrcoef(prepared.to_numpy().T)))[::-1]
    expected = eigenvalues / eigenvalues.sum()
    got = get_explained_variance(model)["explained_variance_ratio"].to_numpy()
    assert got == pytest.approx(expected, rel=1e-8)
    assert eigenvalues.sum() == pytest.approx(N_FEATURES)


# ======================= test 3 and 4: output rows and columns =======================
def test_transformed_output_has_one_row_per_input_row(feature_df):
    model = fit_pca(feature_df)
    out = transform_pca(feature_df, model)
    assert len(out) == len(feature_df)
    assert out.shape == (len(feature_df), 2 + 5)
    assert len(transform_pca(feature_df.iloc[:37], model)) == 37


def test_transformed_output_has_the_expected_columns_and_keeps_ids_and_index(feature_df):
    shuffled_index = feature_df.copy()
    shuffled_index.index = np.arange(len(feature_df))[::-1] + 1000  # a non-default index
    out = transform_pca(shuffled_index, fit_pca(feature_df))
    assert list(out.columns) == ["date", "symbol", "PC1", "PC2", "PC3", "PC4", "PC5"]
    assert out.index.equals(shuffled_index.index)
    assert (out["date"].to_numpy() == shuffled_index["date"].to_numpy()).all()
    assert (out["symbol"].to_numpy() == shuffled_index["symbol"].to_numpy()).all()
    assert np.isfinite(out[pc_columns(out)].to_numpy()).all()


def test_component_scores_are_uncorrelated_on_the_training_data(feature_df):
    out = transform_pca(feature_df, fit_pca(feature_df, n_components=5))
    correlation = np.corrcoef(out[pc_columns(out)].to_numpy().T)
    assert correlation == pytest.approx(np.eye(5), abs=1e-10)  # the defining property of principal components


# ======================= test 5: standardisation happens before PCA =======================
def test_preprocessing_gives_standardised_inputs_even_when_scales_differ_hugely(feature_df):
    prepared = prepare_features(feature_df)
    spreads = prepared.std()
    assert spreads.max() / spreads.min() > 100  # e.g. RSI ~15 vs return_5d ~0.02: wildly different scales

    model = fit_pca(feature_df)
    standardised = preprocess(feature_df, model)
    assert standardised.shape == (len(feature_df), N_FEATURES)
    assert standardised.mean(axis=0) == pytest.approx(0.0, abs=1e-9)
    assert standardised.std(axis=0) == pytest.approx(1.0, abs=1e-9)


def test_pca_is_fitted_on_the_standardised_data_not_the_raw_data(feature_df):
    model = fit_pca(feature_df)
    prepared = prepare_features(feature_df)
    # the scaler learned the means and spreads of the (log-volume) features ...
    assert model.scaler.mean_ == pytest.approx(prepared.mean().to_numpy())
    assert model.scaler.scale_ == pytest.approx(prepared.std(ddof=0).to_numpy())
    # ... and PCA only ever saw centred data (its own mean is zero), so the scaler ran first
    assert model.pca.mean_ == pytest.approx(0.0, abs=1e-9)


def test_without_standardising_one_big_scale_feature_would_dominate(feature_df):
    """Shows WHY we standardise: unscaled, RSI (the largest numbers) swallows the whole analysis."""
    prepared = prepare_features(feature_df)
    unscaled = PCA(n_components=5, svd_solver="full").fit(prepared.to_numpy())
    assert unscaled.explained_variance_ratio_[0] > 0.95
    rsi_position = PCA_FEATURES.index("rsi_14")
    assert abs(unscaled.components_[0][rsi_position]) > 0.99

    scaled = get_explained_variance(fit_pca(feature_df))["explained_variance_ratio"]
    assert scaled.iloc[0] < 0.6
    assert abs(get_feature_loadings(fit_pca(feature_df)).loc["rsi_14", "PC1"]) < 0.6


# ======================= test 6: volume is log-transformed =======================
def test_volume_log_transform_is_applied(feature_df):
    prepared = prepare_features(feature_df)
    np.testing.assert_allclose(prepared[VOLUME_FEATURE], np.log1p(feature_df[VOLUME_FEATURE]), rtol=0, atol=0)
    other_columns = [c for c in PCA_FEATURES if c != VOLUME_FEATURE]
    pd.testing.assert_frame_equal(prepared[other_columns], feature_df[other_columns].astype("float64"))  # others untouched


def test_the_log_transform_really_reduces_the_skew(feature_df):
    raw_skew = feature_df[VOLUME_FEATURE].skew()
    log_skew = prepare_features(feature_df)[VOLUME_FEATURE].skew()
    assert raw_skew > 1.0  # raw volume is strongly right-skewed (a few huge values)
    assert abs(log_skew) < 0.5 * raw_skew


def test_the_scaler_learns_from_log_volume_not_raw_volume(feature_df):
    model = fit_pca(feature_df)
    position = PCA_FEATURES.index(VOLUME_FEATURE)
    assert model.scaler.mean_[position] == pytest.approx(np.log1p(feature_df[VOLUME_FEATURE]).mean())
    assert model.scaler.mean_[position] < 30  # a log is small; raw volume would give a mean above 1,000,000


def test_zero_volume_is_safe(feature_df):
    feature_df.loc[0, VOLUME_FEATURE] = 0.0
    prepared = prepare_features(feature_df)
    assert prepared.loc[0, VOLUME_FEATURE] == 0.0  # log1p(0) = 0, not minus infinity
    assert np.isfinite(prepared.to_numpy()).all()
    fit_pca(feature_df)  # still fits


def test_negative_volume_is_rejected(feature_df):
    feature_df.loc[3, VOLUME_FEATURE] = -5.0
    with pytest.raises(ValueError, match="negative"):
        prepare_features(feature_df)


# ======================= test 7: fit / transform separation =======================
def split_by_time(df, n_train_dates=60):
    cutoff = np.sort(df["date"].unique())[n_train_dates - 1]
    return df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()


def test_transforming_new_data_uses_the_frozen_model_and_learns_nothing(feature_df):
    train, test = split_by_time(feature_df)
    model = fit_pca(train)
    frozen = (model.scaler.mean_.copy(), model.scaler.scale_.copy(), model.pca.components_.copy(), model.pca.mean_.copy())

    out = transform_pca(test, model)

    # nothing inside the model changed
    for before, after in zip(frozen, (model.scaler.mean_, model.scaler.scale_, model.pca.components_, model.pca.mean_)):
        np.testing.assert_array_equal(before, after)

    # the result is exactly "training scaler, then training loadings", worked out by hand
    X = prepare_features(test).to_numpy()
    z = (X - frozen[0]) / frozen[1]
    expected = (z - frozen[3]) @ frozen[2].T
    np.testing.assert_allclose(out[model.component_names].to_numpy(), expected, rtol=1e-9, atol=1e-12)

    # and it is NOT what you would get by refitting on the new data
    refit = transform_pca(test, fit_pca(test))
    assert not np.allclose(out[model.component_names].to_numpy(), refit[model.component_names].to_numpy(), atol=1e-3)

    # transforming again gives the identical answer
    pd.testing.assert_frame_equal(out, transform_pca(test, model))


def test_model_remembers_which_period_it_was_trained_on(feature_df):
    train, _ = split_by_time(feature_df)
    model = fit_pca(train)
    assert model.fit_start == train["date"].min()
    assert model.fit_end == train["date"].max()
    assert model.n_training_rows == len(train)


def test_standardising_new_data_does_not_re_centre_it(feature_df):
    """On later data the preprocessed values are NOT forced to mean 0 / spread 1; they use the training scaler."""
    train, test = split_by_time(feature_df)
    shifted = test.copy()
    shifted["return_20d"] = shifted["return_20d"] + 0.5  # a big shift away from the training average
    z = preprocess(shifted, fit_pca(train))
    assert z[:, PCA_FEATURES.index("return_20d")].mean() > 3  # still far from 0: it was not re-centred


# ======================= test 8: no future leakage =======================
SCORES = ["PC1", "PC2", "PC3", "PC4", "PC5"]


def test_changing_future_observations_does_not_change_earlier_pca_values(feature_df):
    train, _ = split_by_time(feature_df, n_train_dates=60)
    cutoff = train["date"].max()
    future = feature_df["date"] > cutoff

    model = fit_pca(train)
    before = transform_pca(feature_df, model)

    changed = feature_df.copy()
    changed.loc[future, PCA_FEATURES] = changed.loc[future, PCA_FEATURES].to_numpy() * 5.0  # wild future data
    model_after = fit_pca(changed[changed["date"] <= cutoff])  # fitted on the training period only
    after = transform_pca(changed, model_after)

    # the model learned the same thing, because its training rows did not change
    np.testing.assert_array_equal(model.scaler.mean_, model_after.scaler.mean_)
    np.testing.assert_array_equal(model.pca.components_, model_after.pca.components_)
    # so every earlier observation has exactly the same PCA values
    pd.testing.assert_frame_equal(before[~future], after[~future])
    # and the test is sensitive: the changed future rows really did move
    assert not np.allclose(before.loc[future, SCORES], after.loc[future, SCORES])


def test_changing_one_future_observation_changes_only_that_observation(feature_df):
    train, _ = split_by_time(feature_df)
    model = fit_pca(train)
    before = transform_pca(feature_df, model)

    changed = feature_df.copy()
    last = changed.index[-1]
    changed.loc[last, "beta_60d"] = 50.0  # one absurd future value
    after = transform_pca(changed, model)

    pd.testing.assert_frame_equal(before.drop(index=last), after.drop(index=last))  # every other row identical
    assert not np.allclose(before.loc[last, SCORES].to_numpy(dtype=float), after.loc[last, SCORES].to_numpy(dtype=float))


def test_fitting_on_all_the_data_WOULD_leak_the_future(feature_df):
    """
    The wrong way, shown on purpose. If PCA is fitted on everything, changing future rows changes the
    PCA values of EARLIER dates. This is exactly what the backtest must avoid, and why the test above passes.
    """
    train, _ = split_by_time(feature_df)
    future = feature_df["date"] > train["date"].max()
    changed = feature_df.copy()
    changed.loc[future, PCA_FEATURES] = changed.loc[future, PCA_FEATURES].to_numpy() * 5.0

    leaky_before = transform_pca(feature_df, fit_pca(feature_df))
    leaky_after = transform_pca(changed, fit_pca(changed))
    assert not np.allclose(leaky_before.loc[~future, SCORES], leaky_after.loc[~future, SCORES], atol=1e-3)


# ======================= test 9: loadings =======================
def test_loadings_have_the_expected_shape_and_labels(feature_df):
    model = fit_pca(feature_df, n_components=5)
    loadings = get_feature_loadings(model)
    assert loadings.shape == (N_FEATURES, 5)
    assert list(loadings.index) == PCA_FEATURES
    assert list(loadings.columns) == ["PC1", "PC2", "PC3", "PC4", "PC5"]
    np.testing.assert_array_equal(loadings.to_numpy(), model.pca.components_.T)


def test_loadings_are_unit_length_orthogonal_weights(feature_df):
    loadings = get_feature_loadings(fit_pca(feature_df, n_components=5)).to_numpy()
    assert (np.abs(loadings) <= 1 + 1e-12).all()
    assert (loadings**2).sum(axis=0) == pytest.approx(1.0)  # each component's squared weights add to 1
    assert loadings.T @ loadings == pytest.approx(np.eye(5), abs=1e-10)  # components are at right angles


def test_a_component_is_the_weighted_sum_of_the_standardised_features(feature_df):
    """The viva sentence, as a test: PC score = sum of (loading x standardised feature)."""
    model = fit_pca(feature_df, n_components=5)
    scores = transform_pca(feature_df, model)
    standardised = preprocess(feature_df, model)
    loadings = get_feature_loadings(model).to_numpy()
    np.testing.assert_allclose(scores[model.component_names].to_numpy(), standardised @ loadings, rtol=1e-9, atol=1e-12)


def test_top_loadings_are_ranked_by_size_and_keep_their_sign():
    loadings = pd.DataFrame({"PC1": [0.1, -0.8, 0.5, -0.2]}, index=["a", "b", "c", "d"])
    assert top_loadings(loadings, "PC1", k=3) == [("b", -0.8), ("c", 0.5), ("d", -0.2)]
    assert len(top_loadings(loadings, "PC1", k=2)) == 2


# ======================= known-answer test =======================
def test_one_hidden_factor_gives_one_component_with_equal_weights():
    """
    Build all 13 columns from ONE factor z (some with a negative slope). After standardising they are
    identical up to sign, so PC1 must explain 100% and every loading must have size 1/sqrt(13).
    """
    rng = np.random.default_rng(5)
    z = rng.standard_normal(300)
    slopes = np.array([0.02, 0.05, 0.10, -0.03, 0.04, 0.03, 0.05, 15.0, -0.04, 0.3, 0.1, 1.0, 0.2])
    df = pd.DataFrame({"date": pd.bdate_range("2023-01-02", periods=300), "symbol": "X"})
    for name, slope in zip(PCA_FEATURES, slopes):
        df[name] = 1.0 + slope * z
    df[VOLUME_FEATURE] = np.exp(10 + z) - 1  # so that log1p(volume) = 10 + z, linear in z like the others

    model = fit_pca(df, n_components=3)
    assert model.pca.explained_variance_ratio_[0] == pytest.approx(1.0, abs=1e-9)
    loadings = get_feature_loadings(model)["PC1"].to_numpy()
    assert np.abs(loadings) == pytest.approx(1 / np.sqrt(13), rel=1e-6)
    # features with opposite slopes get opposite-sign loadings
    assert np.sign(loadings * loadings[0]) == pytest.approx(np.sign(slopes * slopes[0]))


# ======================= test 10: the original table is not modified =======================
def test_original_feature_table_is_never_modified(feature_df):
    original = feature_df.copy(deep=True)
    model = fit_pca(feature_df)
    prepare_features(feature_df)
    preprocess(feature_df, model)
    out = transform_pca(feature_df, model)
    out.iloc[:, 2:] = 0.0  # even scribbling on the output must not touch the input
    pd.testing.assert_frame_equal(feature_df, original)
    assert (feature_df[VOLUME_FEATURE] == original[VOLUME_FEATURE]).all()  # raw volume, not log volume
    assert feature_df[VOLUME_FEATURE].min() > 1000


def test_column_order_of_the_input_does_not_matter(feature_df):
    reordered = feature_df[["symbol", "date"] + PCA_FEATURES[::-1] + ["close"]]
    model_a, model_b = fit_pca(feature_df), fit_pca(reordered)
    np.testing.assert_allclose(model_a.pca.components_, model_b.pca.components_, atol=1e-12)
    pd.testing.assert_frame_equal(transform_pca(feature_df, model_a), transform_pca(reordered, model_b))


# ======================= bad input =======================
def test_bad_inputs_are_rejected_with_clear_messages(feature_df):
    with pytest.raises(ValueError, match="missing column"):
        fit_pca(feature_df.drop(columns=["beta_60d"]))
    with pytest.raises(ValueError, match="empty"):
        fit_pca(feature_df.iloc[0:0])

    with_nan = feature_df.copy()
    with_nan.loc[2, "rsi_14"] = np.nan
    with pytest.raises(ValueError, match="clean_feature_data"):
        fit_pca(with_nan)

    with_inf = feature_df.copy()
    with_inf.loc[2, "beta_60d"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        fit_pca(with_inf)

    model = fit_pca(feature_df)
    with pytest.raises(ValueError, match="symbol"):
        transform_pca(feature_df.drop(columns=["symbol"]), model)
    with pytest.raises(ValueError, match="missing column"):
        transform_pca(feature_df.drop(columns=["rsi_14"]), model)


# ======================= saving the result =======================
def test_pca_output_is_saved_and_can_be_read_back(feature_df, tmp_path):
    out = transform_pca(feature_df, fit_pca(feature_df))
    path = save_pca_output(out, tmp_path / "nested" / "folder" / "pca_features.csv")  # folders are created
    assert path.exists()
    back = pd.read_csv(path, parse_dates=["date"])
    assert list(back.columns) == list(out.columns)
    assert len(back) == len(out)
    assert (back["date"].to_numpy() == pd.to_datetime(out["date"]).to_numpy()).all()
    assert (back["symbol"] == out["symbol"].to_numpy()).all()
    np.testing.assert_allclose(back[SCORES].to_numpy(), out[SCORES].to_numpy(), rtol=1e-9, atol=1e-12)


# ======================= plots =======================
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_explained_variance_plot_is_written(feature_df, tmp_path):
    model = fit_pca(feature_df)
    path = plot_explained_variance(model, tmp_path / "plots" / "variance.png", note="test note")
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    assert path.stat().st_size > 5_000


def test_scatter_plot_draws_a_sample_and_leaves_the_pca_result_alone(feature_df, tmp_path):
    model = fit_pca(feature_df)
    out = transform_pca(feature_df, model)
    snapshot = out.copy(deep=True)
    drawn = plot_pc_scatter(out, model, tmp_path / "scatter.png", max_points=120)
    assert drawn == 120
    assert (tmp_path / "scatter.png").read_bytes()[:8] == PNG_SIGNATURE
    pd.testing.assert_frame_equal(out, snapshot)  # plotting must not change the PCA data
    assert len(out) == 400

    assert plot_pc_scatter(out, model, tmp_path / "all.png", max_points=10_000) == len(out)  # small data: draw everything


def test_sampling_for_plots_is_repeatable_and_returns_a_copy(feature_df):
    out = transform_pca(feature_df, fit_pca(feature_df))
    a, b = sample_for_plot(out, max_points=50, seed=1), sample_for_plot(out, max_points=50, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not sample_for_plot(out, max_points=50, seed=2).index.equals(a.index)
    assert len(sample_for_plot(out, max_points=1000)) == len(out)
    before = out.copy(deep=True)
    a["PC1"] = 0.0
    pd.testing.assert_frame_equal(out, before)


# ======================= report text =======================
def test_interpretation_comes_from_the_actual_loadings(feature_df):
    model = fit_pca(feature_df, n_components=3)
    text = interpretation_text(model, top_k=3)
    loadings = get_feature_loadings(model)
    assert "=== PCA INTERPRETATION ===" in text
    for name in model.component_names:
        section = text.split(f"{name} (")[1].split("\n\n")[0]
        for feature, value in top_loadings(loadings, name, 3):
            assert feature in section and f"{value:+.3f}" in section
    assert "should not be interpreted as direct predictions of future returns" in text
    assert "momentum" not in text.lower()  # we never hard-code what a component "means"


def test_report_prints_the_computed_numbers_not_invented_ones(feature_df, capsys):
    model = fit_pca(feature_df, n_components=5)
    print_pca_report(model, n_valid_rows=len(feature_df), source="dev_prices_synthetic.csv")
    out = capsys.readouterr().out
    table = get_explained_variance(model)

    for heading in ["=== PCA ANALYSIS ===", "Explained variance:", "Cumulative explained variance:", "=== PCA INTERPRETATION ==="]:
        assert heading in out
    assert "Number of input features: 13" in out
    assert "Number of PCA components: 5" in out
    assert f"Number of valid rows: {len(feature_df):,}" in out
    assert "EXPLORATORY" in out and "look-ahead" in out
    assert "synthetic" in out.lower()

    shown = [float(v) for v in re.findall(r"^PC\d: (\d+\.\d+)%$", out, flags=re.M)]
    expected = list(table["explained_variance_ratio"] * 100) + list(table["cumulative_explained_variance"] * 100)
    assert shown == pytest.approx([round(v, 2) for v in expected], abs=0.006)
    total = table["cumulative_explained_variance"].iloc[-1] * 100
    assert f"Total variance retained by 5 components: {total:.2f}%" in out


# ======================= config =======================
def test_config_has_the_pca_settings():
    cfg = load_config(PROJECT_ROOT / "config.yaml")["pca"]
    assert isinstance(cfg["n_components"], int) and 1 <= cfg["n_components"] <= N_FEATURES
    assert cfg["output_file"].endswith(".csv") and cfg["plots_dir"]


# ======================= the development data, end to end =======================
@pytest.fixture(scope="module")
def dev_feature_data():
    config = load_config(PROJECT_ROOT / "config.yaml")
    stocks = get_provider(config, PROJECT_ROOT).get_historical_data()
    market = LocalDataProvider(PROJECT_ROOT / config["data"]["market_index_file"]).get_historical_data()
    return clean_feature_data(build_features(stocks, market))


def test_pca_runs_on_the_phase2_output(dev_feature_data):
    original = dev_feature_data.copy(deep=True)
    model = fit_pca(dev_feature_data, n_components=5)
    out = transform_pca(dev_feature_data, model)
    assert len(out) == len(dev_feature_data)
    assert list(out.columns) == ["date", "symbol", "PC1", "PC2", "PC3", "PC4", "PC5"]
    assert np.isfinite(out[SCORES].to_numpy()).all()
    assert 0.5 < get_explained_variance(model)["cumulative_explained_variance"].iloc[-1] <= 1
    pd.testing.assert_frame_equal(dev_feature_data, original)  # Phase 2 output untouched


def test_pca_recovers_structure_that_was_built_into_the_synthetic_data(dev_feature_data):
    """
    The synthetic stocks were created with known betas and known trading volumes. If PCA finds real
    structure, some component's per-stock averages should line up with each of them. (Sign is arbitrary,
    so we use the size of the rank correlation.) This checks the pipeline, not any market behaviour.
    """
    model = fit_pca(dev_feature_data, n_components=5)
    averages = transform_pca(dev_feature_data, model).groupby("symbol")[model.component_names].mean()
    designed = pd.DataFrame(
        {"beta": {s: v[2] for s, v in DEV_STOCKS.items()}, "volume": {s: v[5] for s, v in DEV_STOCKS.items()}}
    ).loc[averages.index]

    for trait in ("beta", "volume"):
        correlations = [abs(averages[pc].corr(designed[trait], method="spearman")) for pc in model.component_names]
        assert max(correlations) > 0.9, f"no component lines up with the designed {trait}: {np.round(correlations, 2)}"


def test_command_line_demo_writes_data_and_plots(tmp_path, monkeypatch, capsys):
    config = {
        "data": {
            "provider": "local",
            "prices_file": str(PROJECT_ROOT / "data/raw/dev_prices_synthetic.csv"),
            "market_index_file": str(PROJECT_ROOT / "data/raw/dev_market_index_synthetic.csv"),
            "symbols": ["RELIANCE", "TCS", "HDFCBANK", "INFY"],
            "start_date": None,
            "end_date": None,
        },
        "pca": {"n_components": 5, "output_file": str(tmp_path / "out" / "pca.csv"), "plots_dir": str(tmp_path / "plots")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["pca_model", "--config", str(config_path), "--components", "3"])  # override: 3, not 5

    main()
    text = capsys.readouterr().out

    saved = pd.read_csv(tmp_path / "out" / "pca.csv")
    assert list(saved.columns) == ["date", "symbol", "PC1", "PC2", "PC3"]
    assert set(saved["symbol"]) == {"RELIANCE", "TCS", "HDFCBANK", "INFY"}
    assert (tmp_path / "plots" / "pca_explained_variance.png").exists()
    assert (tmp_path / "plots" / "pca_pc1_pc2_scatter.png").exists()
    for expected in ["=== PCA ANALYSIS ===", "Number of PCA components: 3", "Output shape:", "PCA data saved to:", "Plots saved to:"]:
        assert expected in text
