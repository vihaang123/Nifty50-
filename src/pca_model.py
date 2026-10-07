"""
Phase 3: PCA (Principal Component Analysis) on the 13 Phase 2 features.

What PCA does, in plain words
-----------------------------
Our 13 features overlap a lot (20-day and 60-day volatility move together, the
5/20/60-day returns are related, and so on). PCA replaces them with a few NEW columns,
the principal components (PC1, PC2, ...). Each component is a weighted combination of
the original features:

    PC1 = w1 * feature_1 + w2 * feature_2 + ... + w13 * feature_13

The weights are chosen so that PC1 captures as much of the overall variation in the
data as one column can, PC2 captures as much of what is LEFT as it can (while being
uncorrelated with PC1), and so on. The weights are called loadings.
PCA is unsupervised: it never looks at any label or at future returns.

PCA does not predict prices and does not by itself improve returns. It only describes
how the historical behaviour features are structured.

The pipeline (this order matters)
---------------------------------
    13 features
        |  select the 13 PCA feature columns
        |  log1p(avg_volume_20d)       <- volume only, on a COPY of the data
        |  StandardScaler              <- every column to mean 0, spread 1
        |  PCA
    PC1 ... PCk

Why log the volume? avg_volume_20d is a raw scale variable and is highly skewed: it is
in the millions and a few very busy stocks sit far above the rest. The logarithm shrinks
those extreme values before standardising. We use log1p(x) = log(1 + x) because it is
safe at 0 (log(0) would be minus infinity). The original avg_volume_20d column in the
Phase 2 table is never changed.

Why standardise? The features live on very different scales: volatility is a few tens of
percent, RSI runs 0 to 100, beta sits around 1, volume is in the millions. PCA looks for
directions of largest VARIANCE, so without standardising, the feature with the biggest
numbers would dominate every component just because of its units. StandardScaler rescales
each feature to mean 0 and standard deviation 1, so each feature starts with an equal vote.

LEAKAGE RULE (read this before Phase 7, backtesting)
----------------------------------------------------
The scaler (its means and spreads) and PCA (its weights) are LEARNED from data. They must be
learned only from the data given to fit_pca() and then reused unchanged by transform_pca().
transform_pca() never re-learns anything.

The Phase 3 demo is an EXPLORATORY PCA analysis: it fits on the complete synthetic dataset,
which is acceptable for looking at the structure. The later backtesting pipeline must fit
the preprocessing and PCA only on the training period to avoid look-ahead bias, then transform
the later (test) period with that frozen model.

Run it:   python -m src.pca_model            (or: --components 10)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.figure import Figure  # Figure + savefig needs no pyplot, so no global matplotlib backend is changed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import FEATURE_COLUMNS, build_features, clean_feature_data

# The PCA input is exactly the 13 Phase 2 features. No other features are added.
PCA_FEATURES = list(FEATURE_COLUMNS)
VOLUME_FEATURE = "avg_volume_20d"  # the one column that gets log-transformed
DEFAULT_COMPONENTS = 5
ID_COLUMNS = ["date", "symbol"]


# ---------------------------------------------------------------------------
# The fitted model (just a container for what was learned)
# ---------------------------------------------------------------------------
@dataclass
class PCAModel:
    """Everything learned by fit_pca(). Treat it as read-only after fitting."""

    feature_columns: list[str]
    scaler: StandardScaler  # learned: the mean and spread of each feature in the TRAINING data
    pca: PCA  # learned: the loadings (weights) of each component
    n_training_rows: int
    fit_start: pd.Timestamp | None  # first/last date of the training data (None if no date column)
    fit_end: pd.Timestamp | None

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    @property
    def component_names(self) -> list[str]:
        return [f"PC{i}" for i in range(1, self.n_components + 1)]


# ---------------------------------------------------------------------------
# Step 1-2: select the features and log-transform the volume
# ---------------------------------------------------------------------------
def prepare_features(feature_data: pd.DataFrame) -> pd.DataFrame:
    """
    Select the 13 PCA features and log-transform avg_volume_20d. Returns a NEW table.

    The table passed in is never modified: the original avg_volume_20d stays as it was.
    PCA cannot handle missing values, so they are an error here (we do not fill them);
    run clean_feature_data() from Phase 2 first.
    """
    missing = [c for c in PCA_FEATURES if c not in feature_data.columns]
    if missing:
        raise ValueError(f"feature_data is missing column(s): {missing}")
    if len(feature_data) == 0:
        raise ValueError("feature_data is empty.")

    X = feature_data[PCA_FEATURES].astype("float64").copy()  # copy: the original must stay untouched

    n_bad = int((~np.isfinite(X.to_numpy())).sum())
    if n_bad:
        raise ValueError(
            f"feature_data has {n_bad} missing or infinite feature value(s). "
            "Run clean_feature_data() first; PCA input must be complete."
        )
    if (X[VOLUME_FEATURE] < 0).any():
        raise ValueError(f"{VOLUME_FEATURE} has negative values; volume cannot be negative.")

    X[VOLUME_FEATURE] = np.log1p(X[VOLUME_FEATURE])
    return X


# ---------------------------------------------------------------------------
# fit / transform (kept strictly separate)
# ---------------------------------------------------------------------------
def fit_pca(feature_data: pd.DataFrame, n_components: int = DEFAULT_COMPONENTS) -> PCAModel:
    """
    LEARN the scaler and PCA from `feature_data` (and only from it).

    LEAKAGE: pass training-period data only when backtesting. Everything learned here
    (feature means/spreads, component weights) comes from these rows and no others.
    """
    n_features = len(PCA_FEATURES)
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise ValueError(f"n_components must be a whole number, got {n_components!r}.")
    if not 1 <= n_components <= n_features:
        raise ValueError(f"n_components must be between 1 and {n_features}, got {n_components}.")

    X = prepare_features(feature_data)
    if len(X) <= n_components:
        raise ValueError(f"Need more than {n_components} rows to fit {n_components} components, got {len(X)}.")

    scaler = StandardScaler().fit(X)
    standardised = scaler.transform(X)
    # svd_solver="full" is exact and repeatable (the default can pick a randomised method).
    pca = PCA(n_components=int(n_components), svd_solver="full").fit(standardised)

    dates = pd.to_datetime(feature_data["date"]) if "date" in feature_data.columns else None
    return PCAModel(
        feature_columns=list(PCA_FEATURES),
        scaler=scaler,
        pca=pca,
        n_training_rows=len(X),
        fit_start=dates.min() if dates is not None else None,
        fit_end=dates.max() if dates is not None else None,
    )


def preprocess(feature_data: pd.DataFrame, model: PCAModel) -> np.ndarray:
    """
    Steps 1-3 for any data, using the model's FROZEN scaler: select, log-volume, standardise.

    The scaler's means and spreads come from the training data, not from `feature_data`.
    (So on new data the result is only approximately mean 0 / spread 1, as it should be.)
    """
    return model.scaler.transform(prepare_features(feature_data))


def transform_pca(feature_data: pd.DataFrame, model: PCAModel) -> pd.DataFrame:
    """
    Convert feature rows into PCA coordinates using an ALREADY FITTED model. Nothing is re-learned.

    Returns a new table with columns date, symbol, PC1 ... PCk, one row per input row, in the
    same order and with the same index. `feature_data` itself is left unchanged.
    """
    missing_ids = [c for c in ID_COLUMNS if c not in feature_data.columns]
    if missing_ids:
        raise ValueError(f"feature_data is missing column(s): {missing_ids}")

    scores = model.pca.transform(preprocess(feature_data, model))
    out = pd.DataFrame(scores, columns=model.component_names, index=feature_data.index)
    out.insert(0, "symbol", feature_data["symbol"].to_numpy())
    out.insert(0, "date", feature_data["date"].to_numpy())
    return out


# ---------------------------------------------------------------------------
# Reading the fitted model
# ---------------------------------------------------------------------------
def get_explained_variance(model: PCAModel) -> pd.DataFrame:
    """
    How much of the total variation each component captures.

    Columns (as fractions, 0.25 = 25%):
      explained_variance_ratio      this component's share of ALL 13 standardised features' variance
      cumulative_explained_variance running total from PC1 up to this component
    """
    ratio = model.pca.explained_variance_ratio_
    return pd.DataFrame(
        {"explained_variance_ratio": ratio, "cumulative_explained_variance": np.cumsum(ratio)},
        index=model.component_names,
    )


def get_feature_loadings(model: PCAModel) -> pd.DataFrame:
    """
    The loadings table: one row per original feature, one column per component.

    A principal component is a weighted combination of the (standardised) original features.
    The loading is that weight:
      * the bigger the size (ignoring the sign), the more that feature contributes to the component;
      * the sign says which way: two features with the same sign push the component the same way,
        opposite signs push it in opposite directions;
      * each component's weights have total squared size 1, so values are between -1 and +1;
      * the overall sign of a component is arbitrary (flipping every sign in a column describes the
        same component), so interpret the PATTERN of signs, not whether PC1 is "positive".
    """
    return pd.DataFrame(model.pca.components_.T, index=model.feature_columns, columns=model.component_names)


def top_loadings(loadings: pd.DataFrame, component: str, k: int = 3) -> list[tuple[str, float]]:
    """The k features with the largest loading (by size, ignoring sign) on one component."""
    column = loadings[component]
    order = column.abs().sort_values(ascending=False, kind="stable").index[:k]
    return [(feature, float(column[feature])) for feature in order]


# ---------------------------------------------------------------------------
# Saving and plotting
# ---------------------------------------------------------------------------
def save_pca_output(pca_df: pd.DataFrame, path: str | Path) -> Path:
    """Write the PCA coordinates to a CSV file (creating the folder if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pca_df.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def sample_for_plot(pca_df: pd.DataFrame, max_points: int = 3000, seed: int = 0) -> pd.DataFrame:
    """
    A random subset of rows, for DRAWING ONLY (thousands of overlapping dots are unreadable).
    Returns a copy; the PCA result itself is never reduced or changed.
    """
    if len(pca_df) <= max_points:
        return pca_df.copy()
    return pca_df.sample(n=max_points, random_state=seed)


# Chart colours: one hue for the data, neutral greys for everything else.
_SURFACE, _INK, _INK_2, _MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
_GRID, _AXIS = "#e1e0d9", "#c3c2b7"
_BLUE, _ORANGE = "#2a78d6", "#eb6834"


def _style_axes(ax, grid_axis: str) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", color=_INK, fontsize=12, fontweight="bold", pad=26)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=_INK_2, fontsize=9, va="bottom")


def _label_centres(ax, centres: pd.DataFrame) -> None:
    """
    Label each stock-average dot with its symbol without the labels overlapping.

    The averages often sit close together, so the labels go in one tidy column just to the right
    of the group, ordered top to bottom like the dots, each joined to its dot by a thin line.
    """
    ax.autoscale_view()
    y_low, y_high = ax.get_ylim()
    x_low, x_high = ax.get_xlim()
    min_gap = 0.045 * (y_high - y_low)  # about one line of 9pt text
    label_x = centres["PC1"].max() + 0.06 * (x_high - x_low)

    ordered = centres.sort_values("PC2", ascending=False)
    previous_y = None
    for symbol, row in ordered.iterrows():
        label_y = row["PC2"] if previous_y is None else min(row["PC2"], previous_y - min_gap)
        previous_y = label_y
        label = ax.annotate(
            symbol, xy=(row["PC1"], row["PC2"]), xytext=(label_x, label_y), textcoords="data",
            fontsize=9, color=_INK, va="center", ha="left", zorder=4,
            arrowprops={"arrowstyle": "-", "color": _MUTED, "linewidth": 0.8, "shrinkA": 1, "shrinkB": 5},
        )
        label.set_path_effects([patheffects.withStroke(linewidth=3, foreground=_SURFACE)])  # readable over dots


def plot_explained_variance(model: PCAModel, path: str | Path, note: str | None = None) -> Path:
    """Plot 1: bar chart of the variance each component explains (cumulative total under each label)."""
    table = get_explained_variance(model)
    pct = table["explained_variance_ratio"].to_numpy() * 100
    cumulative = table["cumulative_explained_variance"].to_numpy() * 100
    x = np.arange(len(pct))

    fig = Figure(figsize=(7.5, 4.6), facecolor=_SURFACE)
    ax = fig.subplots()
    _style_axes(ax, grid_axis="y")
    ax.bar(x, pct, width=0.5, color=_BLUE)
    for xi, value in zip(x, pct):
        ax.text(xi, value + pct.max() * 0.015, f"{value:.1f}%", ha="center", va="bottom", color=_INK, fontsize=10)
    ax.set_xticks(x, [f"{name}\ncum. {c:.1f}%" for name, c in zip(table.index, cumulative)])
    ax.set_ylim(0, pct.max() * 1.15)
    ax.set_ylabel("Explained variance (%)", color=_INK_2, fontsize=9)
    _title(
        ax,
        "Variance explained by each principal component",
        f"{len(pct)} components keep {cumulative[-1]:.1f}% of the variation in the 13 standardised features",
    )
    if note:
        fig.text(0.01, 0.005, note, color=_MUTED, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    return path


def plot_pc_scatter(
    pca_df: pd.DataFrame,
    model: PCAModel,
    path: str | Path,
    max_points: int = 3000,
    seed: int = 0,
    note: str | None = None,
) -> int:
    """
    Plot 2: PC1 vs PC2. Each faint dot is one stock-day (a random sample, for readability only).
    Each orange dot is a stock's AVERAGE position over all its observations, labelled with the symbol.
    Returns the number of observation dots drawn.
    """
    sample = sample_for_plot(pca_df, max_points, seed)
    centres = pca_df.groupby("symbol")[["PC1", "PC2"]].mean()  # from ALL rows, not the sample
    ratio = model.pca.explained_variance_ratio_

    fig = Figure(figsize=(8, 6.4), facecolor=_SURFACE)
    ax = fig.subplots()
    _style_axes(ax, grid_axis="both")
    ax.scatter(
        sample["PC1"], sample["PC2"], s=8, color=_BLUE, alpha=0.22, linewidths=0,
        label=f"One stock-day (random sample of {len(sample):,} of {len(pca_df):,})",
    )
    ax.scatter(
        centres["PC1"], centres["PC2"], s=64, color=_ORANGE, edgecolors=_SURFACE, linewidths=1.5, zorder=3,
        label="Stock average over all its days",
    )
    if len(centres) <= 30:  # with hundreds of stocks, labels would just be clutter
        _label_centres(ax, centres)
    ax.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}% of variance)", color=_INK_2, fontsize=9)
    ax.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}% of variance)", color=_INK_2, fontsize=9)
    _title(ax, "Stock-days in the PC1 / PC2 plane", "Distance in this plane shows similarity of historical behaviour, not of future returns")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False, fontsize=9, labelcolor=_INK_2)
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
def interpretation_text(model: PCAModel, top_k: int = 3) -> str:
    """
    A short reading of the loadings. Everything printed comes from the fitted model:
    no meaning is assumed in advance (e.g. we never just declare "PC1 is momentum").
    """
    loadings = get_feature_loadings(model)
    ratios = get_explained_variance(model)["explained_variance_ratio"]
    lines = ["=== PCA INTERPRETATION ===", ""]
    for name in model.component_names:
        lines.append(f"{name} ({ratios[name] * 100:.2f}% of variance) is most influenced by:")
        for rank, (feature, value) in enumerate(top_loadings(loadings, name, top_k), start=1):
            lines.append(f"{rank}. {feature}  (loading {value:+.3f})")
        lines.append("")
    lines += [
        "How to read this: features in a component with the same sign move together within it,",
        "opposite signs move against each other. The overall sign of a component is arbitrary.",
        "",
        "These components describe combinations of historical financial behaviour.",
        "They should not be interpreted as direct predictions of future returns.",
    ]
    return "\n".join(lines)


def print_pca_report(model: PCAModel, n_valid_rows: int, source: str = "") -> None:
    variance = get_explained_variance(model)
    loadings = get_feature_loadings(model)

    print("=== PCA ANALYSIS ===")
    print()
    if "synthetic" in source.lower():
        print("NOTE: synthetic development data (random numbers, NOT real prices).")
        print("Nothing below says anything about real stocks.")
    print("EXPLORATORY: PCA is fitted on the complete dataset here. The later backtesting")
    print("pipeline must fit the preprocessing and PCA only on the training period (no look-ahead).")
    print()
    print(f"Number of input features: {len(model.feature_columns)}")
    print(f"Number of valid rows: {n_valid_rows:,}")
    print(f"Number of PCA components: {model.n_components}")
    print(f"Preprocessing: log1p({VOLUME_FEATURE}) on a copy -> StandardScaler -> PCA")
    print()
    print("Explained variance:")
    print()
    for name, ratio in variance["explained_variance_ratio"].items():
        print(f"{name}: {ratio * 100:.2f}%")
    print()
    print("Cumulative explained variance:")
    print()
    for name, ratio in variance["cumulative_explained_variance"].items():
        print(f"{name}: {ratio * 100:.2f}%")
    print()
    total = variance["cumulative_explained_variance"].iloc[-1]
    print(f"Total variance retained by {model.n_components} components: {total * 100:.2f}%")
    print(f"Variance not captured: {(1 - total) * 100:.2f}%")
    print()
    print("Feature loadings (weight of each standardised feature in each component):")
    print()
    table = loadings.copy()
    table.index.name = "Feature"
    print(table.to_string(float_format=lambda v: f"{v:+.3f}"))
    print()
    print(interpretation_text(model))


# ---------------------------------------------------------------------------
# Command-line demonstration
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory PCA on the Phase 2 features.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--components", type=int, default=None, help="number of components (overrides config.yaml)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    config = load_config(config_path)
    data_cfg, pca_cfg = config["data"], config.get("pca", {})
    n_components = args.components or pca_cfg.get("n_components", DEFAULT_COMPONENTS)
    output_file = base_dir / pca_cfg.get("output_file", "data/processed/pca_features.csv")
    plots_dir = base_dir / pca_cfg.get("plots_dir", "data/processed/plots")

    # Phase 1 -> Phase 2: load prices, build features, drop rows without a full feature set.
    window = {"start": data_cfg.get("start_date"), "end": data_cfg.get("end_date")}
    provider = get_provider(config, base_dir=base_dir)
    stocks = provider.get_historical_data(symbols=data_cfg.get("symbols"), **window)
    market = LocalDataProvider(base_dir / data_cfg["market_index_file"]).get_historical_data(**window)
    feature_data = clean_feature_data(build_features(stocks, market))  # the original feature table stays as it is

    # Phase 3: fit (exploratory: on everything), then transform.
    source = str(getattr(provider, "path", ""))
    model = fit_pca(feature_data, n_components=n_components)
    pca_df = transform_pca(feature_data, model)
    print_pca_report(model, n_valid_rows=len(feature_data), source=source)

    note = "Synthetic development data (random numbers, not real prices)" if "synthetic" in source.lower() else None
    csv_path = save_pca_output(pca_df, output_file)
    bars_path = plot_explained_variance(model, plots_dir / "pca_explained_variance.png", note=note)
    drawn = plot_pc_scatter(pca_df, model, plots_dir / "pca_pc1_pc2_scatter.png", note=note)

    print()
    print(f"Output shape: {pca_df.shape}  (columns: {', '.join(pca_df.columns)})")
    print(f"PCA data saved to: {csv_path}")
    print(f"Plots saved to: {plots_dir}")
    print(f"  - {bars_path.name}")
    print(f"  - {Path(plots_dir / 'pca_pc1_pc2_scatter.png').name}  ({drawn:,} of {len(pca_df):,} observations drawn; for readability only)")


if __name__ == "__main__":
    main()
