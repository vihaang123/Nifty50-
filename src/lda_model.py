"""
Phase 4b: LDA (Linear Discriminant Analysis) on the 13 features, using the constructed classes.

What LDA does, in plain words
-----------------------------
LDA is a SUPERVISED dimensionality-reduction technique that finds the directions that separate
PREDEFINED classes as much as possible. We give it the 13 standardised features and a class for each
observation (Defensive / Balanced / Aggressive, built in src/labels.py), and it returns new columns
LD1 and LD2, each a weighted combination of the features:

    LD1 = c1 * feature_1 + c2 * feature_2 + ... + c13 * feature_13      (after centring)

chosen so that observations of different classes land far apart while observations of the same class
land close together.

PCA vs LDA (two different views of the SAME 13 features)
    PCA  unsupervised, never sees classes, finds directions of maximum VARIANCE.
    LDA  supervised, needs predefined classes, finds directions of maximum CLASS SEPARATION.
LDA does NOT discover the classes. We create them first, with a transparent rule.

Why exactly two components (LD1, LD2)?
With K classes LDA can produce at most K - 1 discriminant directions. We have K = 3 classes, so the
maximum is 2. (Asking sklearn for 3 raises an error; a test checks this.)

The pipeline
------------
    feature data -> behaviour labels (src/labels.py)
                 -> select the 13 features, log1p(avg_volume_20d) on a copy, StandardScaler
                    (exactly the preprocessing of Phase 3, so PCA and LDA see identical inputs)
                 -> LDA -> LD1, LD2

LIMITATION: CIRCULAR BY CONSTRUCTION (read this)
------------------------------------------------
The classes are built from volatility_60d, beta_60d and return_60d, and LDA uses those same features
(plus ten others). So LDA is demonstrating whether the selected features can separate the categories we
constructed from them. It is NOT proving that LDA found natural stock categories, and a high accuracy
is expected. Also, our classes are slices of one ordered score (Defensive < Balanced < Aggressive), so
we expect almost all of the separation to sit on LD1 and very little on LD2.

LDA also assumes each class is roughly bell-shaped with a similar spread. Slices of one distribution
satisfy that only roughly, which is another reason not to over-interpret the numbers.

LEAKAGE RULE
------------
The scaler and the LDA directions are LEARNED from the data given to fit_lda(), and the label rules are
learned from the training data too. transform_lda() re-learns nothing.

The Phase 4 demo is EXPLORATORY, so it learns everything from the complete dataset. A backtest must do:

    training period -> fit scaler -> create training labels (rules learned on the training period)
                    -> fit LDA -> transform FUTURE observations with the frozen model

and never refit using future observations.

Run it:   python -m src.lda_model
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.figure import Figure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import build_features, clean_feature_data
from src.labels import (
    CLASS_NAMES,
    LABEL_COLUMN,
    LABEL_FEATURES,
    MIN_CLASS_SHARE,
    LabelRules,
    check_label_balance,
    create_behavior_labels,
    describe_label_rules,
    fit_label_rules,
    label_counts,
)

# Phase 3 supplies the shared preprocessing (select the 13 features, log-transform volume on a copy),
# the top-contributor helper, the plot sampling and the chart style, so both analyses stay consistent.
from src.pca_model import (  # noqa: E402
    PCA_FEATURES,
    VOLUME_FEATURE,
    _BLUE,
    _INK,
    _INK_2,
    _MUTED,
    _ORANGE,
    _SURFACE,
    _style_axes,
    _title,
    prepare_features,
    sample_for_plot,
    top_loadings,
)

N_LDA_COMPONENTS = 2  # number of classes (3) minus 1: the most LDA can give
LD_COLUMNS = ["LD1", "LD2"]
ID_COLUMNS = ["date", "symbol"]

# One colour per class, kept the same in every chart from now on (slots 1-3 of the project palette,
# checked together for colour-blind safety). Cool = defensive, warm = aggressive.
CLASS_COLOURS = {"Defensive": _BLUE, "Balanced": "#1baf7a", "Aggressive": _ORANGE}


# ---------------------------------------------------------------------------
# The fitted model (just a container for what was learned)
# ---------------------------------------------------------------------------
@dataclass
class LDAModel:
    """Everything learned by fit_lda(). Treat it as read-only after fitting."""

    feature_columns: list[str]
    scaler: StandardScaler  # learned: the mean and spread of each feature in the TRAINING data
    lda: LinearDiscriminantAnalysis  # learned: the discriminant directions and class statistics
    n_training_rows: int
    fit_start: pd.Timestamp | None
    fit_end: pd.Timestamp | None

    @property
    def n_components(self) -> int:
        return len(LD_COLUMNS)

    @property
    def component_names(self) -> list[str]:
        return list(LD_COLUMNS)


# ---------------------------------------------------------------------------
# fit / transform (kept strictly separate)
# ---------------------------------------------------------------------------
def _check_labels(labels: pd.Series, feature_data: pd.DataFrame) -> pd.Series:
    """The labels as plain text, after checking they match the rows and contain all three classes."""
    if not isinstance(labels, pd.Series):
        raise ValueError("labels must be a pandas Series (see create_behavior_labels).")
    if len(labels) != len(feature_data) or not labels.index.equals(feature_data.index):
        raise ValueError("labels must cover exactly the rows of feature_data (same index).")
    if labels.isna().any():
        raise ValueError("labels contain missing values.")
    text = labels.astype(str)
    unknown = sorted(set(text) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown class name(s) in labels: {unknown}. Expected: {CLASS_NAMES}")
    absent = [name for name in CLASS_NAMES if name not in set(text)]
    if absent:
        raise ValueError(f"LDA needs all three classes in the training data; missing: {absent}")
    return text


def fit_lda(feature_data: pd.DataFrame, labels: pd.Series, min_class_share: float = MIN_CLASS_SHARE) -> LDAModel:
    """
    LEARN the scaler and the LDA directions from `feature_data` and its `labels` (and only from them).

    LEAKAGE: when backtesting, pass training-period data only, with labels made from rules learned on the
    training period.
    """
    X = prepare_features(feature_data)  # select the 13 features, log1p volume on a copy (the input is untouched)
    y = _check_labels(labels, feature_data)
    check_label_balance(y, min_class_share)  # refuses an unusably small class

    scaler = StandardScaler().fit(X)
    standardised = scaler.transform(X)
    lda = LinearDiscriminantAnalysis(solver="svd", n_components=N_LDA_COMPONENTS).fit(standardised, y.to_numpy())

    dates = pd.to_datetime(feature_data["date"]) if "date" in feature_data.columns else None
    return LDAModel(
        feature_columns=list(PCA_FEATURES),
        scaler=scaler,
        lda=lda,
        n_training_rows=len(X),
        fit_start=dates.min() if dates is not None else None,
        fit_end=dates.max() if dates is not None else None,
    )


def preprocess_lda(feature_data: pd.DataFrame, model: LDAModel) -> np.ndarray:
    """Select, log-transform volume and standardise, using the model's FROZEN scaler (learned on training data)."""
    return model.scaler.transform(prepare_features(feature_data))


def transform_lda(feature_data: pd.DataFrame, model: LDAModel, labels: pd.Series | None = None) -> pd.DataFrame:
    """
    Project feature rows onto LD1 and LD2 using an ALREADY FITTED model. Nothing is re-learned.

    Returns a new table: date, symbol, [behavior_class,] LD1, LD2 - one row per input row, in the same order
    and with the same index. behavior_class is included only if `labels` is passed (labels are not needed to
    transform; future observations may not have any). `feature_data` is left unchanged.
    """
    missing_ids = [c for c in ID_COLUMNS if c not in feature_data.columns]
    if missing_ids:
        raise ValueError(f"feature_data is missing column(s): {missing_ids}")

    scores = model.lda.transform(preprocess_lda(feature_data, model))[:, :N_LDA_COMPONENTS]
    out = pd.DataFrame(scores, columns=LD_COLUMNS, index=feature_data.index)
    if labels is not None:
        if len(labels) != len(feature_data) or not labels.index.equals(feature_data.index):
            raise ValueError("labels must cover exactly the rows of feature_data (same index).")
        out.insert(0, LABEL_COLUMN, labels.astype(str).to_numpy())
    out.insert(0, "symbol", feature_data["symbol"].to_numpy())
    out.insert(0, "date", feature_data["date"].to_numpy())
    return out


def predict_lda(feature_data: pd.DataFrame, model: LDAModel) -> pd.Series:
    """The class LDA predicts for each row (an ordered categorical, like the labels). Nothing is re-learned."""
    predicted = model.lda.predict(preprocess_lda(feature_data, model))
    return pd.Series(pd.Categorical(predicted, categories=CLASS_NAMES, ordered=True), index=feature_data.index, name="predicted_class")


# ---------------------------------------------------------------------------
# Reading the fitted model
# ---------------------------------------------------------------------------
def get_lda_loadings(model: LDAModel) -> pd.DataFrame:
    """
    The LDA coefficient table: one row per original feature, one column per discriminant (LD1, LD2).

    LD1 = sum of (coefficient x standardised feature), after subtracting the training mean. These are the
    discriminant directions (sklearn's `scalings_`), not the classifier's per-class coefficients (`coef_`).

    Larger absolute coefficients indicate that the feature contributes more strongly to the corresponding
    discriminant direction, although interpretation should consider the sign and scaling:
      * the features are standardised first, so the coefficients are comparable across features;
      * correlated features (for example volatility_20d and volatility_60d) can share or trade off a
        contribution, so one feature's coefficient alone can mislead;
      * the overall sign of a discriminant is arbitrary, so read the pattern of signs, not the sign of LD1.
    """
    coefficients = model.lda.scalings_[:, :N_LDA_COMPONENTS]
    return pd.DataFrame(coefficients, index=model.feature_columns, columns=LD_COLUMNS)


def get_lda_explained_variance(model: LDAModel) -> pd.Series:
    """
    The share of the between-class separation carried by each discriminant (fractions, sum <= 1).
    With classes that form one ordered scale, expect LD1 to carry almost all of it.
    """
    return pd.Series(model.lda.explained_variance_ratio_[:N_LDA_COMPONENTS], index=LD_COLUMNS, name="between_class_share")


def classification_diagnostics(feature_data: pd.DataFrame, labels: pd.Series, model: LDAModel) -> dict:
    """
    A very simple diagnostic: class counts, accuracy, and where the mistakes are.

    CAUTION: when called on the data the model was fitted on, the accuracy is a TRAINING accuracy. It is only a
    diagnostic of how well the features reproduce the constructed classes, not an out-of-sample performance measure.

    Returns a dict with: counts, accuracy, majority_baseline (the score of always guessing the largest class),
    confusion (rows = constructed class, columns = LDA prediction).
    """
    actual = _check_labels(labels, feature_data)
    predicted = predict_lda(feature_data, model).astype(str)
    counts = label_counts(actual)
    confusion = pd.crosstab(
        pd.Categorical(actual, categories=CLASS_NAMES),
        pd.Categorical(predicted, categories=CLASS_NAMES),
        rownames=["Constructed class"],
        colnames=["LDA prediction"],
        dropna=False,
    ).reindex(index=CLASS_NAMES, columns=CLASS_NAMES, fill_value=0)
    return {
        "counts": counts,
        "accuracy": float((actual.to_numpy() == predicted.to_numpy()).mean()),
        "majority_baseline": float(counts.max() / counts.sum()),
        "confusion": confusion,
    }


# ---------------------------------------------------------------------------
# Saving and plotting
# ---------------------------------------------------------------------------
def save_lda_output(lda_df: pd.DataFrame, path: str | Path) -> Path:
    """Write the LDA coordinates and classes to a CSV file (creating the folder if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lda_df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def plot_lda_scatter(
    lda_df: pd.DataFrame,
    model: LDAModel,
    path: str | Path,
    max_points: int = 3000,
    seed: int = 0,
    note: str | None = None,
) -> int:
    """
    LD1 vs LD2, coloured by the constructed class. Each faint dot is one stock-day (a random sample, for
    readability only); each ringed dot is a class's average position over ALL observations, labelled.
    Returns the number of observation dots drawn.
    """
    sample = sample_for_plot(lda_df, max_points, seed)
    classes = lda_df[LABEL_COLUMN].astype(str)
    counts = label_counts(lda_df[LABEL_COLUMN])
    share = get_lda_explained_variance(model)

    fig = Figure(figsize=(8, 6.4), facecolor=_SURFACE)
    ax = fig.subplots()
    _style_axes(ax, grid_axis="both")
    for name in CLASS_NAMES:  # drawn from Defensive to Aggressive
        points = sample[sample[LABEL_COLUMN].astype(str) == name]
        ax.scatter(points["LD1"], points["LD2"], s=9, color=CLASS_COLOURS[name], alpha=0.35, linewidths=0,
                   label=f"{name} ({counts[name]:,} stock-days)")
    for name in CLASS_NAMES:
        rows = lda_df[classes == name]
        x, y = rows["LD1"].mean(), rows["LD2"].mean()
        ax.scatter([x], [y], s=90, color=CLASS_COLOURS[name], edgecolors=_SURFACE, linewidths=1.8, zorder=3)
        label = ax.annotate(name, (x, y), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10,
                            color=_INK, fontweight="bold", zorder=4)
        label.set_path_effects([patheffects.withStroke(linewidth=3, foreground=_SURFACE)])

    ax.set_aspect("equal", adjustable="datalim")  # LD1 and LD2 share one unit, so distances can be compared
    ax.set_xlabel(f"LD1 ({share['LD1'] * 100:.1f}% of between-class separation)", color=_INK_2, fontsize=9)
    ax.set_ylabel(f"LD2 ({share['LD2'] * 100:.1f}% of between-class separation)", color=_INK_2, fontsize=9)
    _title(ax, "Stock-days in the LD1 / LD2 plane",
           "Constructed behavioural classes from the same features LDA uses (not ground truth)")
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=3, frameon=False, fontsize=9, labelcolor=_INK_2)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
        handle.set_sizes([40])
    if note:
        fig.text(0.01, 0.005, note, color=_MUTED, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    return len(sample)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def contributors_text(model: LDAModel, top_k: int = 3) -> str:
    """The strongest contributors to each discriminant. Taken from the fitted model, never hard-coded."""
    loadings = get_lda_loadings(model)
    lines = []
    for name in LD_COLUMNS:
        lines.append(f"{name} strongest contributors:")
        for rank, (feature, value) in enumerate(top_loadings(loadings, name, top_k), start=1):
            lines.append(f"{rank}. {feature}  (coefficient {value:+.3f})")
        lines.append("")
    return "\n".join(lines).rstrip()


def print_lda_report(
    model: LDAModel, feature_data: pd.DataFrame, labels: pd.Series, rules: LabelRules, source: str = ""
) -> dict:
    diagnostics = classification_diagnostics(feature_data, labels, model)
    counts, total = diagnostics["counts"], diagnostics["counts"].sum()
    share = get_lda_explained_variance(model)
    table = get_lda_loadings(model)
    table.index.name = "Feature"

    print("=== LDA ANALYSIS ===")
    print()
    if "synthetic" in source.lower():
        print("NOTE: The current dataset is synthetic and is used only to verify that the implementation works.")
        print("      Nothing below says anything about real stocks or real markets.")
    print("EXPLORATORY: labels rules, scaler and LDA are learned from the complete dataset here. A backtest must learn")
    print("them from the training period only, then transform later observations with the frozen model.")
    print("CIRCULAR BY CONSTRUCTION: the classes are built from volatility_60d, beta_60d and return_60d, which LDA also")
    print("uses. LDA shows whether the chosen features can separate the categories WE constructed. It is not discovering")
    print("natural stock categories, and a high accuracy is expected.")
    print()
    print(describe_label_rules(rules))
    print()
    print(f"Number of observations: {len(feature_data):,}")
    print(f"Input features: {len(model.feature_columns)}")
    print(f"Classes: {len(CLASS_NAMES)}")
    print(f"LDA components: {model.n_components}   (at most classes - 1 = {len(CLASS_NAMES) - 1})")
    print()
    print("Class distribution:")
    for name in CLASS_NAMES:
        print(f"{name}: {counts[name]:,}  ({counts[name] / total:.1%})")
    print()
    print(f"Training accuracy: {diagnostics['accuracy'] * 100:.2f}%")
    print()
    print("Share of the between-class separation carried by each discriminant:")
    for name in LD_COLUMNS:
        print(f"{name}: {share[name] * 100:.2f}%")
    print()
    print(contributors_text(model))
    print()
    print("LDA coefficients (one per standardised feature; LD = sum of coefficient x feature):")
    print()
    print(table.to_string(float_format=lambda v: f"{v:+.3f}"))
    print()
    print("Larger absolute coefficients indicate that the feature contributes more strongly to the corresponding")
    print("discriminant direction, although interpretation should consider the sign and scaling. Correlated features can")
    print("share or trade off a contribution, and the overall sign of LD1 / LD2 is arbitrary.")
    print()
    print("=== LDA CLASSIFICATION ===")
    print()
    for name in CLASS_NAMES:
        print(f"{name}: {counts[name]:,}")
    print()
    print(f"Training accuracy: {diagnostics['accuracy'] * 100:.2f}%")
    print(f"(For scale: always guessing the largest class would score {diagnostics['majority_baseline'] * 100:.2f}%.)")
    print()
    print("Confusion table (rows = constructed class, columns = LDA prediction):")
    print(diagnostics["confusion"].to_string())
    print()
    print("NOTE:")
    print("Training accuracy is only a diagnostic because the model was evaluated")
    print("on the same observations used for fitting.")
    print("It is not an out-of-sample performance measure.")
    return diagnostics


# ---------------------------------------------------------------------------
# Command-line demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory LDA on the Phase 2 features and constructed classes.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    config = load_config(config_path)
    data_cfg, lda_cfg = config["data"], config.get("lda", {})
    output_file = base_dir / lda_cfg.get("output_file", "data/processed/lda_features.csv")
    plot_file = base_dir / lda_cfg.get("plot_file", "data/processed/plots/lda_scatter.png")

    # Phase 1 -> Phase 2: load prices, build features, drop rows without a full feature set.
    window = {"start": data_cfg.get("start_date"), "end": data_cfg.get("end_date")}
    provider = get_provider(config, base_dir=base_dir)
    stocks = provider.get_historical_data(symbols=data_cfg.get("symbols"), **window)
    market = LocalDataProvider(base_dir / data_cfg["market_index_file"]).get_historical_data(**window)
    feature_data = clean_feature_data(build_features(stocks, market))  # this table is never modified below

    # Phase 4: rules -> labels -> LDA. Exploratory, so everything is learned from the full dataset.
    rules = fit_label_rules(feature_data)
    labels = create_behavior_labels(feature_data, rules)
    model = fit_lda(feature_data, labels)
    lda_df = transform_lda(feature_data, model, labels)

    source = str(getattr(provider, "path", ""))
    print_lda_report(model, feature_data, labels, rules, source=source)

    note = "Synthetic development data (random numbers, not real prices)" if "synthetic" in source.lower() else None
    csv_path = save_lda_output(lda_df, output_file)
    drawn = plot_lda_scatter(lda_df, model, plot_file, note=note)

    print()
    print(f"Output shape: {lda_df.shape}  (columns: {', '.join(lda_df.columns)})")
    print(f"LDA data saved to: {csv_path}")
    print(f"Plot saved to: {plot_file}  ({drawn:,} of {len(lda_df):,} observations drawn; for readability only)")


if __name__ == "__main__":
    main()
