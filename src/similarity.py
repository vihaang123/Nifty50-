"""
Phase 5: stock behavioural similarity (cosine similarity in PCA space).

The question
------------
    "Given a stock, which other stocks have behaved most similarly, according to the
     financial features used in this project?"

This is NOT price prediction, NOT return prediction, NOT a buy/sell signal, NOT a recommendation
system, and NOT proof that similar stocks will perform similarly in the future.

How it works
------------
    PCA output (one row per stock per day: PC1 ... PC5)
        -> one profile per stock: the MEAN of each PC across that stock's rows
        -> cosine similarity between every pair of profiles
        -> for a chosen stock, the other stocks ranked from most to least similar

"For exploratory similarity analysis, each stock is represented by the mean of its PCA component
scores across its available historical observations."

Cosine similarity in plain words
--------------------------------
    cosine_similarity(A, B) = (A . B) / (||A|| x ||B||)

It compares the DIRECTION two profiles point in and ignores how long they are:
    1.0  very similar direction     0.0  unrelated direction     -1.0  opposite directions
So a profile and the same profile scaled by 3 score exactly 1.0. Two things follow:
  * PCA scores are centred (their average over ALL stock-days is 0), so a stock's mean profile is the
    stock's DEVIATION from the typical stock-day. Cosine similarity then asks whether two stocks deviate
    in the same direction, not by how much.
  * A profile of all zeros has no direction, so cosine similarity is undefined for it. We reject it with
    an error instead of silently returning a made-up number.

What similarity does NOT mean
-----------------------------
Not higher expected return, not lower risk, not a better investment, not future correlation, not
guaranteed diversification. 0.95 between stock A and stock B means their historical PCA profiles point in
very similar directions. It does NOT mean B will return 95% of what A returns.

LEAKAGE LIMITATION (read this)
------------------------------
The current profiles use the full historical period of each stock. That is acceptable for exploratory
analysis, but future backtesting must construct stock profiles using only information available up to the
relevant rebalance date. The PCA scores themselves must also come from a PCA fitted on the training period
only (Phase 3). This similarity engine is NOT suitable for live trading as it stands.

Run it:   python -m src.similarity            (add --symbol SYMBOL --top-n 5 to choose)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_config
from src.pca_model import _BLUE, _INK, _INK_2, _MUTED, _SURFACE, _style_axes, _title

DISCLAIMER = "Similarity represents historical behavioural similarity in PCA space. It does not represent expected future returns."
DEFAULT_TOP_N = 5


# ---------------------------------------------------------------------------
# Stock profiles: one row per stock
# ---------------------------------------------------------------------------
def pc_columns(table: pd.DataFrame) -> list[str]:
    """The principal-component columns (PC1, PC2, ...) of a table, in numerical order (PC2 before PC10)."""
    found = [c for c in table.columns if re.fullmatch(r"PC\d+", str(c))]
    return sorted(found, key=lambda c: int(c[2:]))


def create_stock_profiles(pca_data: pd.DataFrame) -> pd.DataFrame:
    """
    One behavioural profile per stock: the mean of each PC across all of that stock's rows.

    Returns a new table with columns symbol, PC1, ..., PCk and exactly one row per symbol (sorted by symbol).
    No dates. `pca_data` is left unchanged.

    Missing or infinite PC values are rejected, not skipped: skipping would quietly average different
    stocks over different sets of days.
    """
    if "symbol" not in pca_data.columns:
        raise ValueError("pca_data needs a 'symbol' column.")
    components = pc_columns(pca_data)
    if not components:
        raise ValueError("pca_data has no principal-component columns (PC1, PC2, ...). Run Phase 3 first.")
    if len(pca_data) == 0:
        raise ValueError("pca_data is empty.")
    values = pca_data[components].apply(pd.to_numeric, errors="coerce").astype("float64")
    if not np.isfinite(values.to_numpy()).all():
        bad = pca_data.loc[~np.isfinite(values).all(axis=1), "symbol"].astype(str).unique().tolist()
        raise ValueError(f"pca_data contains missing, non-numeric or infinite PC values (symbols affected: {bad}).")
    if pca_data["symbol"].isna().any():
        raise ValueError("pca_data has rows without a symbol.")

    profiles = values.groupby(pca_data["symbol"].astype(str).to_numpy(), sort=True).mean()
    profiles.index.name = "symbol"
    return profiles.reset_index()


def _validated_profiles(stock_profiles: pd.DataFrame) -> tuple[list[str], np.ndarray, list[str]]:
    """Check a profile table is fit for cosine similarity. Returns (symbols, matrix of PC values, PC columns)."""
    if "symbol" not in stock_profiles.columns:
        raise ValueError("stock_profiles needs a 'symbol' column (see create_stock_profiles).")
    components = pc_columns(stock_profiles)
    if not components:
        raise ValueError("stock_profiles has no principal-component columns (PC1, PC2, ...).")
    if len(stock_profiles) == 0:
        raise ValueError("stock_profiles is empty.")
    symbols = stock_profiles["symbol"].astype(str).tolist()
    if len(set(symbols)) != len(symbols):
        duplicated = sorted({s for s in symbols if symbols.count(s) > 1})
        raise ValueError(f"stock_profiles must have one row per symbol; repeated: {duplicated}")
    matrix = stock_profiles[components].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(matrix).all():
        raise ValueError("stock_profiles contains missing, non-numeric or infinite values; similarity would be wrong.")
    flat = [s for s, row in zip(symbols, matrix) if not row.any()]
    if flat:
        raise ValueError(f"Cosine similarity is undefined for an all-zero profile (symbols: {flat}).")
    return symbols, matrix, components


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def calculate_similarity(stock_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise cosine similarity between every pair of stock profiles, as a square table (stocks x stocks).

    Rows and columns hold the same symbols, the diagonal is 1, and the table is symmetric.
    Uses sklearn's cosine_similarity. Values are kept inside [-1, 1] (rounding can give 1.0000000000000002).
    """
    symbols, matrix, _ = _validated_profiles(stock_profiles)
    similarity = cosine_similarity(matrix)
    similarity = (similarity + similarity.T) / 2  # remove last-digit asymmetry from floating-point rounding
    similarity = np.clip(similarity, -1.0, 1.0)
    return pd.DataFrame(similarity, index=pd.Index(symbols, name="symbol"), columns=pd.Index(symbols, name="symbol"))


def find_similar_stocks(stock_profiles: pd.DataFrame, symbol: str, top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """
    The `top_n` stocks most similar to `symbol`, best first. The stock itself is never included.

    Returns a table with columns rank (1 = most similar), symbol, similarity. If fewer than `top_n` other
    stocks exist, all of them are returned. Ties are ordered by symbol so results are reproducible.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n < 1:
        raise ValueError(f"top_n must be a positive whole number, got {top_n!r}.")
    similarity = calculate_similarity(stock_profiles)
    if symbol not in similarity.index:
        raise ValueError(f"Unknown symbol {symbol!r}. Available symbols: {', '.join(similarity.index)}")

    others = similarity.loc[symbol].drop(index=symbol)  # a stock is never its own neighbour
    ranked = pd.DataFrame({"symbol": others.index, "similarity": others.to_numpy()})
    ranked = ranked.sort_values(["similarity", "symbol"], ascending=[False, True], kind="mergesort").head(int(top_n))
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Saving and plotting
# ---------------------------------------------------------------------------
def save_similarity_matrix(similarity: pd.DataFrame, path: str | Path) -> Path:
    """Write the similarity matrix to a CSV file (creating the folder if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    similarity.to_csv(path)
    return path


def plot_similar_stocks(similar: pd.DataFrame, symbol: str, path: str | Path, note: str | None = None) -> Path:
    """A horizontal bar chart of the most similar stocks to `symbol` (most similar at the top), values labelled."""
    if len(similar) == 0:
        raise ValueError("There are no similar stocks to plot.")
    values = similar["similarity"].to_numpy()
    names = similar["symbol"].astype(str).tolist()
    positions = np.arange(len(similar))[::-1]  # rank 1 on top

    fig = Figure(figsize=(8, 1.2 + 0.62 * len(similar)), facecolor=_SURFACE)
    ax = fig.subplots()
    _style_axes(ax, grid_axis="x")
    ax.barh(positions, values, height=0.5, color=_BLUE)
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    ax.set_yticks(positions, names, fontsize=10, color=_INK)
    ax.set_xlim(min(0.0, values.min() * 1.15) - 0.02, 1.08)
    for y, value in zip(positions, values):  # labels sit just right of the end of a positive bar, or of the zero line for a negative one
        ax.annotate(f"{value:.4f}", (max(value, 0.0), y), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, color=_INK)
    ax.set_xlabel("Cosine similarity of mean PCA profiles (1 = same direction, 0 = unrelated, -1 = opposite)",
                  color=_INK_2, fontsize=9)
    _title(ax, f"Stocks most similar to {symbol}",
           "Historical behavioural similarity in PCA space, not expected future returns")
    if note:
        fig.text(0.01, 0.005, note, color=_MUTED, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Report and command-line demo
# ---------------------------------------------------------------------------
def print_similarity_report(
    stock_profiles: pd.DataFrame, symbol: str, similar: pd.DataFrame, matrix_path: str | Path, source: str = ""
) -> None:
    components = pc_columns(stock_profiles)
    print("=== STOCK BEHAVIOURAL SIMILARITY ===")
    print()
    if "synthetic" in source.lower():
        print("NOTE: The current dataset is synthetic and is used only to verify that the implementation works.")
        print("      Nothing below says anything about real stocks or real markets.")
    print("EXPLORATORY: each stock is represented by the mean of its PCA component scores across its available")
    print("historical observations (the full period). Backtesting must use only information available up to each")
    print("rebalance date.")
    print()
    print(f"PCA components used: {len(components)}")
    print(f"Number of stocks: {len(stock_profiles)}")
    print()
    print(f"Selected stock: {symbol}")
    print()
    print("Most similar stocks:")
    print(f"{'Rank':<6}{'Symbol':<12}{'Similarity'}")
    for row in similar.itertuples(index=False):
        print(f"{row.rank:<6}{row.symbol:<12}{row.similarity:.4f}")
    print()
    print("Similarity matrix:")
    print(matrix_path)
    print()
    print(DISCLAIMER)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory stock similarity from the Phase 3 PCA output.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--symbol", default=None, help="stock to find neighbours for (default: first symbol available)")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="how many similar stocks to show (default 5)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    config = load_config(config_path)
    sim_cfg = config.get("similarity", {})
    pca_file = base_dir / config.get("pca", {}).get("output_file", "data/processed/pca_features.csv")
    matrix_file = base_dir / sim_cfg.get("matrix_file", "data/processed/similarity_matrix.csv")
    plots_dir = base_dir / sim_cfg.get("plots_dir", "data/processed/plots")

    if not pca_file.exists():
        parser.error(f"PCA output not found at {pca_file}. Run 'python -m src.pca_model' first.")
    pca_data = pd.read_csv(pca_file)  # the Phase 3 output; never modified below

    profiles = create_stock_profiles(pca_data)
    similarity = calculate_similarity(profiles)
    matrix_path = save_similarity_matrix(similarity, matrix_file)

    symbol = args.symbol if args.symbol is not None else profiles["symbol"].iloc[0]
    try:
        similar = find_similar_stocks(profiles, symbol, args.top_n)
    except ValueError as error:
        parser.error(str(error))

    source = str(config.get("data", {}).get("prices_file", ""))
    print_similarity_report(profiles, symbol, similar, matrix_path, source=source)

    note = "Synthetic development data (random numbers, not real prices)" if "synthetic" in source.lower() else None
    plot_path = plot_similar_stocks(similar, symbol, plots_dir / f"similarity_{symbol}.png", note=note)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
