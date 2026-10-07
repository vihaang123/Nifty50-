"""
Tests for Phase 4 (src/labels.py and src/lda_model.py).

Most tests use small made-up tables, some of them hand-computable, so a wrong answer is obvious.
A few use the project's synthetic development data end to end. They check that the PIPELINE is
correct; they say nothing about real markets (the data is random).

Run:  pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import FEATURE_COLUMNS, build_features, clean_feature_data
from src.labels import (
    CLASS_NAMES,
    LABEL_COLUMN,
    LABEL_FEATURES,
    MIN_CLASS_SHARE,
    behavior_score,
    check_label_balance,
    create_behavior_labels,
    describe_label_rules,
    fit_label_rules,
    label_counts,
)
from src.lda_model import (
    CLASS_COLOURS,
    LD_COLUMNS,
    classification_diagnostics,
    contributors_text,
    fit_lda,
    get_lda_explained_variance,
    get_lda_loadings,
    main,
    plot_lda_scatter,
    predict_lda,
    print_lda_report,
    save_lda_output,
    transform_lda,
)
from src.pca_model import PCA_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
N_FEATURES = 13


# ======================= helpers =======================
def make_feature_df(per_symbol=100, seed=0, symbols=("AAA", "BBB", "CCC", "DDD")):
    """A made-up Phase 2 style table (date, symbol, close + 13 features) with realistic scales."""
    rng = np.random.default_rng(seed)
    n = per_symbol * len(symbols)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)

    def noise(scale):
        return scale * rng.standard_normal(n)

    return pd.DataFrame(
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
            "avg_volume_20d": np.exp(14 + rng.standard_normal(n)),
            "volume_change": noise(0.3),
        }
    )


@pytest.fixture
def feature_df():
    return make_feature_df()


def ordered_six_rows():
    """Six rows whose three label features all rise together: the ranks (and so the score) are exactly 1/6 ... 6/6."""
    return pd.DataFrame(
        {
            "volatility_60d": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            "beta_60d": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
            "return_60d": [-0.05, -0.02, 0.00, 0.02, 0.05, 0.08],
        }
    )


def as_text(labels):
    return labels.astype(str).tolist()


def split_by_date(df, fraction=0.6):
    cutoff = df["date"].sort_values().iloc[int(len(df) * fraction)]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff], cutoff


def wreck_the_future(df, cutoff):
    """A copy of df whose rows on/after the cutoff date are replaced by extreme made-up values."""
    changed = df.copy()
    future = changed["date"] >= cutoff
    for col in LABEL_FEATURES:
        changed.loc[future, col] = changed.loc[future, col] * 25 + 5
    changed.loc[future, "rsi_14"] = 99.0
    changed.loc[future, "avg_volume_20d"] = changed.loc[future, "avg_volume_20d"] * 40
    return changed


# ======================= tests 1 and 2: the classes =======================
def test_exactly_three_classes_are_created(feature_df):
    labels = create_behavior_labels(feature_data=feature_df)
    assert labels.nunique() == 3
    assert len(CLASS_NAMES) == 3


def test_labels_are_defensive_balanced_and_aggressive(feature_df):
    labels = create_behavior_labels(feature_df)
    assert CLASS_NAMES == ["Defensive", "Balanced", "Aggressive"]
    assert set(as_text(labels)) == {"Defensive", "Balanced", "Aggressive"}
    assert labels.name == LABEL_COLUMN == "behavior_class"
    assert list(labels.cat.categories) == CLASS_NAMES and labels.cat.ordered  # Defensive < Balanced < Aggressive
    assert labels.isna().sum() == 0


def test_labels_keep_the_input_index_and_do_not_modify_the_input(feature_df):
    shuffled = feature_df.sample(frac=1, random_state=3)
    original = shuffled.copy(deep=True)
    labels = create_behavior_labels(shuffled)
    assert labels.index.equals(shuffled.index)
    pd.testing.assert_frame_equal(shuffled, original)


# ======================= hand-computed examples =======================
def test_labels_on_a_hand_computed_example():
    df = ordered_six_rows()
    rules = fit_label_rules(df)
    # ranks are 1/6 ... 6/6 and the score is their average, so score = [1/6, 2/6, 3/6, 4/6, 5/6, 1]
    assert behavior_score(df, rules).to_numpy() == pytest.approx([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0])
    # cuts = 1/3 and 2/3 percentiles of those six scores (linear interpolation): 0.3333 + 2/3*(1/6) and 4/6 + 1/3*(1/6)
    assert rules.score_cuts == pytest.approx((4 / 9, 13 / 18))
    assert as_text(create_behavior_labels(df, rules)) == ["Defensive", "Defensive", "Balanced", "Balanced", "Aggressive", "Aggressive"]


def test_a_mixed_row_lands_in_the_middle_class():
    """High volatility (rank 1) + low beta (rank 0) + middling return (rank 0.5) averages 0.5, which is Balanced."""
    rules = fit_label_rules(ordered_six_rows())
    new = pd.DataFrame({"volatility_60d": [9.0], "beta_60d": [-9.0], "return_60d": [0.01]})  # 3 of the 6 reference returns are <= 0.01
    assert behavior_score(new, rules).iloc[0] == pytest.approx(0.5)
    assert as_text(create_behavior_labels(new, rules)) == ["Balanced"]


def test_extreme_rows_are_still_labelled_by_the_frozen_rules():
    rules = fit_label_rules(ordered_six_rows())
    new = pd.DataFrame(
        {"volatility_60d": [-5.0, 50.0], "beta_60d": [-5.0, 50.0], "return_60d": [-5.0, 50.0]}
    )
    assert as_text(create_behavior_labels(new, rules)) == ["Defensive", "Aggressive"]


def test_higher_volatility_beta_and_return_make_a_stock_more_aggressive(feature_df):
    rules = fit_label_rules(feature_df)
    base = feature_df.head(1).copy()
    scores = []
    for col in LABEL_FEATURES:  # raise one feature at a time: the score must go up (return counts with a PLUS sign)
        raised = base.copy()
        raised[col] = feature_df[col].max()
        scores.append(behavior_score(raised, rules).iloc[0] >= behavior_score(base, rules).iloc[0])
    assert all(scores)


# ======================= test 3: labels do not use future data =======================
def test_earlier_labels_do_not_change_when_future_rows_change(feature_df):
    train, _, cutoff = split_by_date(feature_df)
    rules = fit_label_rules(train)  # the rules are learned on the training period only
    before = create_behavior_labels(feature_df, rules)
    after = create_behavior_labels(wreck_the_future(feature_df, cutoff), rules)
    earlier = feature_df["date"] < cutoff
    assert as_text(before[earlier]) == as_text(after[earlier])
    assert as_text(before[~earlier]) != as_text(after[~earlier])  # the future rows themselves did change


def test_a_label_depends_only_on_its_own_row_and_the_frozen_rules(feature_df):
    train, test, _ = split_by_date(feature_df)
    rules = fit_label_rules(train)
    together = create_behavior_labels(test, rules)
    one_by_one = pd.concat([create_behavior_labels(test.iloc[[i]], rules) for i in range(0, len(test), 17)])
    assert as_text(together.loc[one_by_one.index]) == as_text(one_by_one)


def test_labels_ignore_columns_that_are_not_label_features(feature_df):
    rules = fit_label_rules(feature_df)
    changed = feature_df.copy()
    for col in ["close", "rsi_14", "avg_volume_20d", "max_drawdown", "volume_change"]:
        changed[col] = changed[col].to_numpy()[::-1]
    assert as_text(create_behavior_labels(changed, rules)) == as_text(create_behavior_labels(feature_df, rules))


def test_learning_the_rules_on_ALL_data_would_leak_the_future(feature_df):
    """The contrast that justifies fit/apply: with rules learned from every row, future rows move earlier labels."""
    _, _, cutoff = split_by_date(feature_df)
    earlier = feature_df["date"] < cutoff
    before = create_behavior_labels(feature_df)  # no rules passed: learned from all rows (exploratory only)
    after = create_behavior_labels(wreck_the_future(feature_df, cutoff))
    assert as_text(before[earlier]) != as_text(after[earlier])


# ======================= test 4: class sizes =======================
def test_classes_are_reasonably_populated(feature_df):
    labels = create_behavior_labels(feature_df)
    counts = check_label_balance(labels)  # raises if any class has under 5%
    assert counts.sum() == len(feature_df)
    assert (counts / counts.sum() > MIN_CLASS_SHARE).all()
    assert counts.max() - counts.min() <= 0.02 * len(labels)  # percentile cuts make the classes nearly equal


def test_a_class_with_3_percent_of_the_rows_fails():
    labels = pd.Series(["Defensive"] * 3 + ["Balanced"] * 48 + ["Aggressive"] * 49)
    with pytest.raises(ValueError, match="Unusable class balance.*Defensive: 3"):
        check_label_balance(labels)


def test_a_missing_class_fails_and_exactly_5_percent_is_allowed():
    with pytest.raises(ValueError, match="Aggressive: 0"):
        check_label_balance(pd.Series(["Defensive"] * 50 + ["Balanced"] * 50))
    ok = pd.Series(["Defensive"] * 5 + ["Balanced"] * 45 + ["Aggressive"] * 50)
    assert check_label_balance(ok)["Defensive"] == 5


def test_label_counts_are_in_class_order_and_zero_when_absent():
    counts = label_counts(pd.Series(["Aggressive", "Defensive", "Aggressive"]))
    assert list(counts.index) == CLASS_NAMES
    assert counts.tolist() == [1, 0, 2]


# ======================= label input checks =======================
def test_label_functions_reject_bad_input(feature_df):
    with pytest.raises(ValueError, match="missing column"):
        fit_label_rules(feature_df.drop(columns=["beta_60d"]))
    with pytest.raises(ValueError, match="empty"):
        fit_label_rules(feature_df.head(0))
    bad = feature_df.copy()
    bad.loc[bad.index[5], "return_60d"] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        create_behavior_labels(bad)
    bad.loc[bad.index[5], "return_60d"] = np.inf
    with pytest.raises(ValueError, match="missing or infinite"):
        fit_label_rules(bad)


def test_a_score_with_no_variety_cannot_be_split_into_three_classes():
    same = pd.DataFrame({c: [1.0] * 30 for c in LABEL_FEATURES})
    with pytest.raises(ValueError, match="too little variety"):
        fit_label_rules(same)


def test_the_rules_remember_their_training_period_and_describe_themselves(feature_df):
    train, _, _ = split_by_date(feature_df)
    rules = fit_label_rules(train)
    assert rules.n_training_rows == len(train)
    assert rules.fit_start == train["date"].min() and rules.fit_end == train["date"].max()
    text = describe_label_rules(rules)
    assert f"{rules.score_cuts[0]:.3f}" in text and f"{rules.score_cuts[1]:.3f}" in text
    for name in CLASS_NAMES + LABEL_FEATURES:
        assert name in text
    assert fit_label_rules(ordered_six_rows()).fit_start is None  # no date column, no period


# ======================= tests 5 and 6: two components, LD1 and LD2 =======================
def test_lda_produces_exactly_two_components(feature_df):
    model = fit_lda(feature_df, create_behavior_labels(feature_df))
    assert model.n_components == 2
    assert model.component_names == ["LD1", "LD2"] == LD_COLUMNS
    assert model.lda.n_components == 2
    out = transform_lda(feature_df, model)
    assert out[LD_COLUMNS].shape == (len(feature_df), 2)
    assert len(get_lda_explained_variance(model)) == 2


def test_three_discriminants_are_impossible_with_three_classes(feature_df):
    """classes - 1 = 2 is the ceiling; sklearn refuses a third component, which is the viva point."""
    X = feature_df[PCA_FEATURES].to_numpy()
    y = create_behavior_labels(feature_df).astype(str).to_numpy()
    with pytest.raises(ValueError, match="n_components"):
        LinearDiscriminantAnalysis(n_components=3).fit(X, y)


def test_output_has_ld1_and_ld2_in_the_expected_order(feature_df):
    labels = create_behavior_labels(feature_df)
    model = fit_lda(feature_df, labels)
    with_labels = transform_lda(feature_df, model, labels)
    assert list(with_labels.columns) == ["date", "symbol", "behavior_class", "LD1", "LD2"]
    assert list(transform_lda(feature_df, model).columns) == ["date", "symbol", "LD1", "LD2"]
    assert np.isfinite(with_labels[LD_COLUMNS].to_numpy()).all()
    assert as_text(with_labels[LABEL_COLUMN]) == as_text(labels)


# ======================= test 7: row count =======================
def test_output_row_count_index_and_order_match_the_input(feature_df):
    model = fit_lda(feature_df, create_behavior_labels(feature_df))
    subset = feature_df.sample(frac=0.4, random_state=1)  # shuffled, with a gappy index
    out = transform_lda(subset, model)
    assert len(out) == len(subset)
    assert out.index.equals(subset.index)
    assert (out["symbol"].to_numpy() == subset["symbol"].to_numpy()).all()
    assert (out["date"].to_numpy() == subset["date"].to_numpy()).all()
    assert len(transform_lda(feature_df.head(1), model)) == 1


# ======================= test 8: fit / transform separation =======================
def test_transform_uses_the_frozen_model_and_never_refits(feature_df):
    train, test, _ = split_by_date(feature_df)
    model = fit_lda(train, create_behavior_labels(train, fit_label_rules(train)))
    frozen = (model.scaler.mean_.copy(), model.scaler.scale_.copy(), model.lda.scalings_.copy(), model.lda.xbar_.copy())

    shifted = test.copy()  # very different from the training data: a refit would visibly move the model
    shifted["beta_60d"] += 3
    shifted["avg_volume_20d"] *= 50
    out = transform_lda(shifted, model)

    for before, after in zip(frozen, (model.scaler.mean_, model.scaler.scale_, model.lda.scalings_, model.lda.xbar_)):
        np.testing.assert_array_equal(before, after)  # nothing learned from the new rows

    # By hand: log1p the volume, standardise with the TRAINING mean/spread, centre with the TRAINING class-mean average, project.
    X = shifted[PCA_FEATURES].astype("float64").copy()
    X["avg_volume_20d"] = np.log1p(X["avg_volume_20d"])
    z = (X.to_numpy() - frozen[0]) / frozen[1]
    expected = (z - frozen[3]) @ frozen[2][:, :2]
    np.testing.assert_allclose(out[LD_COLUMNS].to_numpy(), expected, atol=1e-9)


def test_a_row_scores_the_same_alone_as_in_a_batch(feature_df):
    model = fit_lda(feature_df, create_behavior_labels(feature_df))
    batch = transform_lda(feature_df, model)
    single = transform_lda(feature_df.iloc[[7]], model)
    np.testing.assert_allclose(single[LD_COLUMNS].to_numpy(), batch.loc[[7], LD_COLUMNS].to_numpy(), atol=1e-9)


# ======================= test 9: the original data is not modified =======================
def test_the_original_feature_table_is_never_modified(feature_df):
    original = feature_df.copy(deep=True)
    rules = fit_label_rules(feature_df)
    labels = create_behavior_labels(feature_df, rules)
    model = fit_lda(feature_df, labels)
    transform_lda(feature_df, model, labels)
    predict_lda(feature_df, model)
    classification_diagnostics(feature_df, labels, model)
    pd.testing.assert_frame_equal(feature_df, original)  # volume is still raw (not log-transformed), nothing added


# ======================= test 10: coefficient dimensions =======================
def test_coefficient_tables_have_the_expected_dimensions(feature_df):
    model = fit_lda(feature_df, create_behavior_labels(feature_df))
    loadings = get_lda_loadings(model)
    assert loadings.shape == (N_FEATURES, 2)
    assert list(loadings.index) == list(PCA_FEATURES) == list(FEATURE_COLUMNS)
    assert list(loadings.columns) == ["LD1", "LD2"]
    assert np.isfinite(loadings.to_numpy()).all()
    assert model.lda.scalings_.shape[0] == N_FEATURES  # the directions: one weight per feature
    assert model.lda.coef_.shape == (3, N_FEATURES)  # the classifier weights: one row per class (a different thing)
    assert model.lda.means_.shape == (3, N_FEATURES)
    assert model.lda.priors_.shape == (3,)


def test_loadings_are_the_discriminant_directions_not_the_classifier_weights(feature_df):
    model = fit_lda(feature_df, create_behavior_labels(feature_df))
    np.testing.assert_array_equal(get_lda_loadings(model).to_numpy(), model.lda.scalings_[:, :2])
    assert not np.allclose(get_lda_loadings(model).to_numpy(), model.lda.coef_[:2].T)


def test_between_class_shares_are_valid_fractions(feature_df):
    share = get_lda_explained_variance(model := fit_lda(feature_df, create_behavior_labels(feature_df)))
    assert list(share.index) == LD_COLUMNS
    assert ((share >= 0) & (share <= 1)).all()
    assert share.sum() == pytest.approx(1.0)  # with 3 classes, 2 discriminants carry all of it
    assert share["LD1"] >= share["LD2"]
    assert model.n_components == 2


# ======================= test 11: no future leakage through the LDA =======================
def test_earlier_lda_output_does_not_change_when_future_rows_change(feature_df):
    train, test, cutoff = split_by_date(feature_df)
    rules = fit_label_rules(train)
    model = fit_lda(train, create_behavior_labels(train, rules))  # fitted on the training period ONLY

    before = transform_lda(feature_df, model)
    after = transform_lda(wreck_the_future(feature_df, cutoff), model)
    earlier = (feature_df["date"] < cutoff).to_numpy()
    np.testing.assert_array_equal(before.loc[earlier, LD_COLUMNS].to_numpy(), after.loc[earlier, LD_COLUMNS].to_numpy())
    assert not np.allclose(before.loc[~earlier, LD_COLUMNS].to_numpy(), after.loc[~earlier, LD_COLUMNS].to_numpy())


def test_refitting_on_all_data_would_leak_the_future_into_earlier_output(feature_df):
    """The contrast: if the model were refitted on every row, earlier rows would be projected differently."""
    train, _, cutoff = split_by_date(feature_df)
    earlier = (feature_df["date"] < cutoff).to_numpy()

    def refit_on_everything(data):
        return transform_lda(data, fit_lda(data, create_behavior_labels(data)))

    before = refit_on_everything(feature_df)
    after = refit_on_everything(wreck_the_future(feature_df, cutoff))
    assert not np.allclose(before.loc[earlier, LD_COLUMNS].to_numpy(), after.loc[earlier, LD_COLUMNS].to_numpy())


def test_the_model_remembers_its_training_period(feature_df):
    train, _, _ = split_by_date(feature_df)
    model = fit_lda(train, create_behavior_labels(train))
    assert model.n_training_rows == len(train)
    assert model.fit_start == train["date"].min() and model.fit_end == train["date"].max()


# ======================= test 12: the three classes are handled correctly =======================
def three_cluster_data(per_class=80, seed=5):
    """Features with a built-in class structure on the three label features; the class names are given directly."""
    rng = np.random.default_rng(seed)
    df = make_feature_df(per_symbol=per_class * 3 // 4 + 1, seed=seed).head(per_class * 3).reset_index(drop=True)
    classes = np.repeat(CLASS_NAMES, per_class)
    level = np.repeat([-1.0, 0.0, 1.0], per_class)
    df["volatility_60d"] = 0.25 + 0.06 * level + 0.01 * rng.standard_normal(len(df))
    df["beta_60d"] = 1.0 + 0.5 * level + 0.08 * rng.standard_normal(len(df))
    df["return_60d"] = 0.02 + 0.08 * level + 0.03 * rng.standard_normal(len(df))
    labels = pd.Series(pd.Categorical(classes, categories=CLASS_NAMES, ordered=True), index=df.index, name=LABEL_COLUMN)
    return df, labels


def test_three_well_separated_classes_are_recovered():
    df, labels = three_cluster_data()
    model = fit_lda(df, labels)
    predicted = predict_lda(df, model)
    assert list(predicted.cat.categories) == CLASS_NAMES and predicted.cat.ordered
    assert (predicted.astype(str) == labels.astype(str)).mean() > 0.97
    diagnostics = classification_diagnostics(df, labels, model)
    assert diagnostics["accuracy"] > 0.97
    assert diagnostics["majority_baseline"] == pytest.approx(1 / 3)
    assert diagnostics["counts"].tolist() == [80, 80, 80]
    assert diagnostics["confusion"].to_numpy().sum() == len(df)
    assert list(diagnostics["confusion"].index) == CLASS_NAMES == list(diagnostics["confusion"].columns)
    assert (np.diag(diagnostics["confusion"].to_numpy()) >= 77).all()


def test_majority_baseline_is_the_share_of_the_largest_class():
    df, _ = three_cluster_data()
    labels = pd.Series(pd.Categorical(["Defensive"] * 120 + ["Balanced"] * 80 + ["Aggressive"] * 40, categories=CLASS_NAMES, ordered=True), index=df.index)
    diagnostics = classification_diagnostics(df, labels, fit_lda(df, labels))
    assert diagnostics["majority_baseline"] == pytest.approx(120 / 240)
    assert diagnostics["counts"].tolist() == [120, 80, 40]


def test_fit_applies_the_log_volume_step_before_standardising(feature_df):
    """The model must be fitted on log1p(volume), the same input it later transforms (shared with PCA)."""
    model = fit_lda(feature_df, create_behavior_labels(feature_df))
    i = PCA_FEATURES.index("avg_volume_20d")
    assert model.scaler.mean_[i] == pytest.approx(np.log1p(feature_df["avg_volume_20d"]).mean())
    assert model.scaler.mean_[PCA_FEATURES.index("rsi_14")] == pytest.approx(feature_df["rsi_14"].mean())


def test_the_class_averages_are_ordered_along_ld1_and_the_priors_are_equal():
    df, labels = three_cluster_data()
    model = fit_lda(df, labels)
    out = transform_lda(df, model, labels)
    means = out.groupby(out[LABEL_COLUMN].astype(str))["LD1"].mean()[CLASS_NAMES]
    ordered = means.is_monotonic_increasing or means.is_monotonic_decreasing  # sign of LD1 is arbitrary, the order is not
    assert ordered and abs(means["Aggressive"] - means["Defensive"]) > 2
    assert abs(means["Balanced"] - means["Defensive"]) > 0.5 and abs(means["Balanced"] - means["Aggressive"]) > 0.5
    np.testing.assert_allclose(model.lda.priors_, [1 / 3] * 3)
    assert sorted(model.lda.classes_) == sorted(CLASS_NAMES)


def test_lda_finds_the_one_feature_that_separates_the_classes():
    """Known answer: only beta_60d differs between the classes; every other column is noise unrelated to them."""
    df = make_feature_df(per_symbol=150, seed=11)
    rng = np.random.default_rng(2)
    labels = pd.Series(pd.Categorical(rng.choice(CLASS_NAMES, len(df)), categories=CLASS_NAMES, ordered=True), index=df.index)
    df["beta_60d"] = 1.0 + 1.2 * labels.cat.codes.to_numpy() + 0.1 * rng.standard_normal(len(df))
    model = fit_lda(df, labels)
    loadings = get_lda_loadings(model)
    assert loadings["LD1"].abs().idxmax() == "beta_60d"
    assert loadings["LD1"].abs()["beta_60d"] > 5 * loadings["LD1"].abs().drop("beta_60d").max()
    assert classification_diagnostics(df, labels, model)["accuracy"] > 0.8


def test_fit_lda_rejects_unusable_labels(feature_df):
    labels = create_behavior_labels(feature_df)
    with pytest.raises(ValueError, match="missing: \\['Aggressive'\\]"):
        fit_lda(feature_df, labels.astype(str).replace("Aggressive", "Balanced"))
    with pytest.raises(ValueError, match="Unknown class"):
        fit_lda(feature_df, labels.astype(str).replace("Balanced", "Neutral"))
    with pytest.raises(ValueError, match="same index"):
        fit_lda(feature_df, labels.iloc[:-1])
    with pytest.raises(ValueError, match="same index"):
        fit_lda(feature_df, labels.reset_index(drop=True).iloc[::-1])
    with pytest.raises(ValueError, match="Series"):
        fit_lda(feature_df, labels.to_numpy())
    tiny_class = labels.astype(str).copy()
    tiny_class.iloc[:] = "Balanced"
    tiny_class.iloc[:6], tiny_class.iloc[6:200] = "Defensive", "Aggressive"  # Defensive = 1.5% of rows
    with pytest.raises(ValueError, match="Unusable class balance"):
        fit_lda(feature_df, tiny_class)


def test_transform_lda_rejects_bad_input(feature_df):
    model = fit_lda(feature_df, create_behavior_labels(feature_df))
    with pytest.raises(ValueError, match="missing column"):
        transform_lda(feature_df.drop(columns=["symbol"]), model)
    with pytest.raises(ValueError, match="same index"):
        transform_lda(feature_df, model, create_behavior_labels(feature_df).iloc[:-1])
    with pytest.raises(ValueError):  # a missing feature column is refused by the shared Phase 3 preparation
        transform_lda(feature_df.drop(columns=["rsi_14"]), model)


# ======================= output and report =======================
def test_saved_csv_round_trips(feature_df, tmp_path):
    labels = create_behavior_labels(feature_df)
    out = transform_lda(feature_df, fit_lda(feature_df, labels), labels)
    path = save_lda_output(out, tmp_path / "nested" / "lda.csv")
    saved = pd.read_csv(path, parse_dates=["date"])
    assert list(saved.columns) == ["date", "symbol", "behavior_class", "LD1", "LD2"]
    assert len(saved) == len(out)
    assert set(saved["behavior_class"]) == set(CLASS_NAMES)
    np.testing.assert_allclose(saved[LD_COLUMNS].to_numpy(), out[LD_COLUMNS].to_numpy(), atol=1e-9)
    assert (saved["date"].to_numpy() == pd.to_datetime(out["date"]).to_numpy()).all()


def test_scatter_plot_draws_a_sample_and_leaves_the_result_alone(feature_df, tmp_path):
    labels = create_behavior_labels(feature_df)
    model = fit_lda(feature_df, labels)
    out = transform_lda(feature_df, model, labels)
    snapshot = out.copy(deep=True)
    drawn = plot_lda_scatter(out, model, tmp_path / "plots" / "lda.png", max_points=120, note="test note")
    assert drawn == 120
    assert (tmp_path / "plots" / "lda.png").read_bytes()[:8] == PNG_SIGNATURE
    pd.testing.assert_frame_equal(out, snapshot)
    assert plot_lda_scatter(out, model, tmp_path / "all.png", max_points=10_000) == len(out)


def test_class_colours_cover_every_class_and_are_distinct():
    assert list(CLASS_COLOURS) == CLASS_NAMES
    assert len(set(CLASS_COLOURS.values())) == 3


def test_report_numbers_come_from_the_fitted_model(feature_df, capsys):
    rules = fit_label_rules(feature_df)
    labels = create_behavior_labels(feature_df, rules)
    model = fit_lda(feature_df, labels)
    diagnostics = print_lda_report(model, feature_df, labels, rules, source="data/raw/dev_prices_synthetic.csv")
    text = capsys.readouterr().out

    fresh = classification_diagnostics(feature_df, labels, model)
    assert diagnostics["accuracy"] == pytest.approx(fresh["accuracy"])
    pd.testing.assert_frame_equal(diagnostics["confusion"], fresh["confusion"])
    counts = label_counts(labels)
    for expected in [
        "=== LDA ANALYSIS ===",
        "=== LDA CLASSIFICATION ===",
        "The current dataset is synthetic and is used only to verify that the implementation works.",
        "CIRCULAR BY CONSTRUCTION",
        "EXPLORATORY",
        f"Number of observations: {len(feature_df):,}",
        "Input features: 13",
        "Classes: 3",
        "LDA components: 2",
        f"Training accuracy: {diagnostics['accuracy'] * 100:.2f}%",
        "Training accuracy is only a diagnostic because the model was evaluated\non the same observations used for fitting.\nIt is not an out-of-sample performance measure.",
        "Confusion table",
    ]:
        assert expected in text
    for name in CLASS_NAMES:
        assert f"{name}: {counts[name]:,}" in text
    assert contributors_text(model) in text
    loadings = get_lda_loadings(model)
    for feature in PCA_FEATURES:  # every feature appears in the loadings table with its fitted LD1 value
        assert f"{loadings.loc[feature, 'LD1']:+.3f}" in text


def test_the_synthetic_notice_only_appears_for_synthetic_data(feature_df, capsys):
    rules = fit_label_rules(feature_df)
    labels = create_behavior_labels(feature_df, rules)
    print_lda_report(fit_lda(feature_df, labels), feature_df, labels, rules, source="data/raw/real_prices.csv")
    assert "synthetic" not in capsys.readouterr().out.lower()


def test_contributors_follow_the_fitted_coefficients(feature_df):
    model = fit_lda(feature_df, create_behavior_labels(feature_df))
    loadings = get_lda_loadings(model)
    text = contributors_text(model, top_k=3)
    for name in LD_COLUMNS:
        strongest = loadings[name].abs().sort_values(ascending=False).index[:3]
        listed = [line.split()[1] for line in text.split(f"{name} strongest contributors:")[1].splitlines()[1:4]]
        assert listed == list(strongest)


# ======================= config =======================
def test_config_has_the_lda_settings():
    cfg = load_config(PROJECT_ROOT / "config.yaml")["lda"]
    assert cfg["output_file"].endswith("lda_features.csv")
    assert cfg["plot_file"].endswith("lda_scatter.png")


# ======================= the development data, end to end =======================
@pytest.fixture(scope="module")
def dev_feature_data():
    config = load_config(PROJECT_ROOT / "config.yaml")
    stocks = get_provider(config, PROJECT_ROOT).get_historical_data()
    market = LocalDataProvider(PROJECT_ROOT / config["data"]["market_index_file"]).get_historical_data()
    return clean_feature_data(build_features(stocks, market))


def test_lda_runs_on_the_phase2_output(dev_feature_data):
    original = dev_feature_data.copy(deep=True)
    labels = create_behavior_labels(dev_feature_data)
    counts = check_label_balance(labels)
    assert counts.tolist() == [counts.iloc[0]] * 3 or counts.max() - counts.min() <= 0.01 * len(labels)

    model = fit_lda(dev_feature_data, labels)
    out = transform_lda(dev_feature_data, model, labels)
    assert out.shape == (len(dev_feature_data), 5)
    assert list(out.columns) == ["date", "symbol", "behavior_class", "LD1", "LD2"]
    assert np.isfinite(out[LD_COLUMNS].to_numpy()).all()
    pd.testing.assert_frame_equal(dev_feature_data, original)  # Phase 2 output untouched

    diagnostics = classification_diagnostics(dev_feature_data, labels, model)
    assert diagnostics["accuracy"] > diagnostics["majority_baseline"] + 0.3  # far above guessing the biggest class
    assert get_lda_explained_variance(model)["LD1"] > 0.9  # one ordered scale, so almost everything sits on LD1


def test_ld1_is_built_from_the_features_the_classes_were_built_from(dev_feature_data):
    """
    EXPECTED, and circular: the classes come from volatility_60d, beta_60d and return_60d, so LD1 must lean on
    those three. This confirms the pipeline is wired correctly; it is not a finding about markets.
    """
    labels = create_behavior_labels(dev_feature_data)
    loadings = get_lda_loadings(fit_lda(dev_feature_data, labels))
    top_three = set(loadings["LD1"].abs().sort_values(ascending=False).index[:3])
    assert top_three == set(LABEL_FEATURES)
    signs = np.sign(loadings.loc[LABEL_FEATURES, "LD1"])
    assert signs.nunique() == 1  # all three push LD1 the same way (the overall sign is arbitrary)


# ======================= command-line demo =======================
def test_command_line_demo_writes_data_and_plot(tmp_path, monkeypatch, capsys):
    config = {
        "data": {
            "provider": "local",
            "prices_file": str(PROJECT_ROOT / "data/raw/dev_prices_synthetic.csv"),
            "market_index_file": str(PROJECT_ROOT / "data/raw/dev_market_index_synthetic.csv"),
            "symbols": ["RELIANCE", "TCS", "HDFCBANK", "INFY"],
            "start_date": None,
            "end_date": None,
        },
        "lda": {"output_file": str(tmp_path / "out" / "lda.csv"), "plot_file": str(tmp_path / "plots" / "lda.png")},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["lda_model", "--config", str(config_path)])

    main()
    text = capsys.readouterr().out

    saved = pd.read_csv(tmp_path / "out" / "lda.csv")
    assert list(saved.columns) == ["date", "symbol", "behavior_class", "LD1", "LD2"]
    assert set(saved["symbol"]) == {"RELIANCE", "TCS", "HDFCBANK", "INFY"}
    assert set(saved["behavior_class"]) == set(CLASS_NAMES)
    assert (tmp_path / "plots" / "lda.png").read_bytes()[:8] == PNG_SIGNATURE
    for expected in ["=== LDA ANALYSIS ===", "=== LDA CLASSIFICATION ===", "Output shape:", "LDA data saved to:", "Plot saved to:"]:
        assert expected in text
