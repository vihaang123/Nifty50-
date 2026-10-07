"""
Phase 6b: a simple, rule-based, multi-cap, behaviourally diversified stock basket.

The idea, in one paragraph (the viva explanation)
-------------------------------------------------
First we divide the stocks into Large, Mid and Small Cap categories. Then we use the LDA behavioural
classes to see whether stocks are Defensive, Balanced or Aggressive. We also use PCA-based similarity to
avoid selecting too many stocks with almost identical historical behaviour. Finally a simple greedy
selection process builds a multi-cap, behaviourally diversified basket, and every stock gets an equal weight.

What this is NOT
----------------
There is no optimiser here (no Markowitz, Sharpe, risk parity, ...), no returns, and no backtest. The
basket is built to provide market-cap and behavioural diversification while reducing excessive similarity
between the selected stocks. It has NOT been tested against any benchmark, so nothing here says it is a
better portfolio. Whether it is better is the question for Phase 7. It is not an investment recommendation.

Inputs
------
    lda_data           Phase 4 output (date, symbol, behavior_class, LD1, LD2). Only symbol and behavior_class
                       are used: each stock's behaviour is its MOST FREQUENT class over its history
                       (ties go to the first of Defensive, Balanced, Aggressive).
    similarity_matrix  Phase 5 output (stock x stock cosine similarity).
    universe           Phase 6a table (symbol, cap_category).

The Phase 2 feature table is deliberately NOT an input. Everything the basket needs from it has already
been summarised by PCA, LDA and the similarity matrix, and an input that nothing reads would only mislead.

The algorithm (greedy, one stock at a time)
-------------------------------------------
1. ELIGIBLE stocks are those in the universe that also have a behaviour label and a row in the similarity
   matrix. Others are left out and reported in the notes.
2. CAP TARGETS: split the basket 40% Large / 30% Mid / 30% Small using the largest-remainder method
   (whole numbers, deterministic; any tie goes to the larger cap first). 10 -> 4/3/3, 6 -> 2/2/2, 15 -> 6/5/4.
3. SUPPLY: if a cap category has fewer eligible stocks than its target, take what exists and give each
   missing slot to the cap category with the most unused eligible stocks (ties: Large, Mid, Small). This is
   reported in the notes. It never crashes and never pretends the target was met.
4. PICK one stock at a time. Candidates are the unselected eligible stocks in cap categories that still have
   room. Each candidate is ranked by, in this order:
       a. not redundant  (similarity with every already-selected stock <= the threshold, default 0.90),
       b. its behaviour class is the least represented in the basket so far,
       c. lowest similarity to the already-selected stocks,
       d. symbol (alphabetical), so the result is always the same.
   Redundancy is a PENALTY, not a ban: if every remaining candidate exceeds the threshold, the least-bad
   one is taken, because the basket still has to reach its size. The notes say so.
5. Equal weights: weight = 1 / basket_size, allocation = capital x weight.

Every pick records WHY it was made (cap slot, behaviour balance, similarity to the basket), and the reason
text is written from those recorded facts, never from a template that ignores them.

LEAKAGE
-------
No returns or prices are used anywhere in selection, so the generator cannot look at the future. The
inputs, however, come from the full synthetic history, which is acceptable for this exploratory demo only.
In a backtest the order must be:

    training data -> features -> PCA fitted on training data -> LDA fitted using training data/rules
    -> similarity profiles built from the training history -> basket generated -> future period evaluated

Run it:   python -m src.basket                                  (add --basket-size 6 --capital 50000)
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from src.data_loader import load_config
from src.labels import CLASS_NAMES
from src.lda_model import CLASS_COLOURS
from src.pca_model import _GRID, _INK, _INK_2, _MUTED, _SURFACE, _style_axes, _title
from src.universe import CAP_CATEGORIES, DEVELOPMENT_NOTICE, check_universe_table, get_development_universe

BASKET_COLUMNS = ["symbol", "cap_category", "behavior_class", "weight", "allocation", "reason"]
MIN_BASKET_SIZE = 3
DEFAULT_BASKET_SIZE = 10
DEFAULT_CAPITAL = 100_000
DEFAULT_SIMILARITY_THRESHOLD = 0.90

# Ideal split of a basket across cap categories, in whole percent (so the arithmetic is exact).
CAP_SHARE_PERCENT = {"Large Cap": 40, "Mid Cap": 30, "Small Cap": 30}

DISCLAIMER = "This basket is generated from synthetic development data and is not an investment recommendation."
METHOD_STATEMENT = (
    "The algorithm constructs a basket designed to provide market-cap and behavioural diversification while "
    "reducing excessive similarity between selected stocks. It has not yet been tested against a benchmark."
)
_SIMILARITY_ROUND = 9  # similarities closer than this count as ties (so floating-point noise cannot change a pick)


@dataclass
class BasketResult:
    """The basket plus what is needed to explain it. `basket` has exactly BASKET_COLUMNS."""

    basket: pd.DataFrame
    target_allocation: dict  # ideal number of stocks per cap category
    planned_allocation: dict  # number per cap category after eligible-supply limits
    notes: list = field(default_factory=list)  # anything that could not be fully satisfied, in plain English
    capital: float = 0.0
    basket_size: int = 0
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    n_eligible: int = 0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def format_rupees(amount: float) -> str:
    """₹100,000 for whole amounts, ₹8,333.33 otherwise."""
    amount = float(amount)
    return f"₹{amount:,.0f}" if abs(amount - round(amount)) < 0.005 else f"₹{amount:,.2f}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))


def _check_capital(capital) -> float:
    if not _is_number(capital) or not math.isfinite(capital) or capital <= 0:
        raise ValueError(f"capital must be a positive number, got {capital!r}.")
    return float(capital)


def _check_basket_size(basket_size) -> int:
    if isinstance(basket_size, (bool, np.bool_)) or not isinstance(basket_size, (int, np.integer)):
        raise ValueError(f"basket_size must be a whole number, got {basket_size!r}.")
    if basket_size < MIN_BASKET_SIZE:
        raise ValueError(f"basket_size must be at least {MIN_BASKET_SIZE}, got {basket_size}.")
    return int(basket_size)


def _check_threshold(threshold) -> float:
    if not _is_number(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError(f"similarity_threshold must be a number between -1 and 1, got {threshold!r}.")
    return float(threshold)


# ---------------------------------------------------------------------------
# Cap targets
# ---------------------------------------------------------------------------
def cap_targets(basket_size: int) -> dict:
    """
    How many stocks to ask for from each cap category (40% Large, 30% Mid, 30% Small), as whole numbers.

    Largest-remainder rounding: give each category the whole-number part of its share, then hand the leftover
    slots, one each, to the categories with the biggest fractional parts. Ties go to the larger cap first.
    Exact integer arithmetic, so the answer never depends on floating-point rounding.
        3 -> 1/1/1   6 -> 2/2/2   10 -> 4/3/3   15 -> 6/5/4
    """
    basket_size = _check_basket_size(basket_size)
    whole = {cap: (basket_size * CAP_SHARE_PERCENT[cap]) // 100 for cap in CAP_CATEGORIES}
    remainder = {cap: (basket_size * CAP_SHARE_PERCENT[cap]) % 100 for cap in CAP_CATEGORIES}
    leftover = basket_size - sum(whole.values())
    by_remainder = sorted(CAP_CATEGORIES, key=lambda cap: (-remainder[cap], CAP_CATEGORIES.index(cap)))
    for cap in by_remainder[:leftover]:
        whole[cap] += 1
    return whole


def plan_cap_quotas(target: dict, available: dict, basket_size: int) -> dict:
    """
    Turn ideal cap targets into feasible quotas given how many eligible stocks each category really has.

    Each category gets min(target, available). Every remaining slot goes to the category with the most unused
    eligible stocks (ties: Large, Mid, Small). Requires sum(available) >= basket_size.
    """
    if sum(available.values()) < basket_size:
        raise ValueError("Not enough eligible stocks to fill the basket.")
    quota = {cap: min(target[cap], available[cap]) for cap in CAP_CATEGORIES}
    while sum(quota.values()) < basket_size:
        spare = {cap: available[cap] - quota[cap] for cap in CAP_CATEGORIES}
        best = max(CAP_CATEGORIES, key=lambda cap: (spare[cap], -CAP_CATEGORIES.index(cap)))
        quota[best] += 1
    return quota


# ---------------------------------------------------------------------------
# Reading the inputs
# ---------------------------------------------------------------------------
def stock_behaviour(lda_data: pd.DataFrame) -> pd.DataFrame:
    """
    One behaviour class per stock: its most frequent class across all of its rows (ties go to the first of
    Defensive, Balanced, Aggressive).

    Returns a table indexed by symbol with behavior_class, share (the fraction of that stock's rows in the
    chosen class) and n_observations. Missing or unknown class names raise an error rather than being skipped.
    """
    missing = [c for c in ("symbol", "behavior_class") if c not in lda_data.columns]
    if missing:
        raise ValueError(f"lda_data is missing column(s): {missing}")
    if len(lda_data) == 0:
        raise ValueError("lda_data is empty.")
    if lda_data["symbol"].isna().any() or lda_data["behavior_class"].isna().any():
        raise ValueError("lda_data has missing symbol or behavior_class values; every stock needs a valid label.")
    labels = lda_data["behavior_class"].astype(str).to_numpy()  # plain arrays: the table's own index (possibly with repeats) is irrelevant
    unknown = sorted(set(labels) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"lda_data has invalid behaviour class(es): {unknown}. Allowed: {CLASS_NAMES}")
    counts = pd.crosstab(lda_data["symbol"].astype(str).to_numpy(), labels).reindex(columns=CLASS_NAMES, fill_value=0)
    table = pd.DataFrame(
        {
            "behavior_class": counts.idxmax(axis=1),  # on a tie, the first column in class order wins
            "share": counts.max(axis=1) / counts.sum(axis=1),
            "n_observations": counts.sum(axis=1),
        }
    )
    table.index.name = "symbol"
    return table


def _check_similarity_matrix(similarity_matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(similarity_matrix, pd.DataFrame) or similarity_matrix.shape[0] == 0:
        raise ValueError("similarity_matrix must be a non-empty square DataFrame (stock x stock).")
    rows, columns = similarity_matrix.index.astype(str), similarity_matrix.columns.astype(str)
    if similarity_matrix.shape[0] != similarity_matrix.shape[1] or sorted(rows) != sorted(columns):
        raise ValueError("similarity_matrix must be square, with the same symbols in its rows and columns.")
    if len(set(rows)) != len(rows):
        raise ValueError("similarity_matrix has repeated symbols.")
    matrix = pd.DataFrame(similarity_matrix.to_numpy(dtype="float64"), index=rows, columns=columns)
    values = matrix.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("similarity_matrix contains missing or infinite values.")
    if values.min() < -1 - 1e-9 or values.max() > 1 + 1e-9:
        raise ValueError("similarity_matrix has values outside [-1, 1]; it is not a cosine similarity matrix.")
    matrix = matrix.loc[sorted(matrix.index), sorted(matrix.index)]  # same order in rows and columns, whatever the input order
    if not np.allclose(matrix.to_numpy(), matrix.to_numpy().T, atol=1e-6):
        raise ValueError("similarity_matrix is not symmetric.")
    return matrix


# ---------------------------------------------------------------------------
# The reason text (built from what actually happened at each pick)
# ---------------------------------------------------------------------------
def _reason(pick: dict, threshold: float) -> str:
    cap, behaviour = pick["cap"], pick["behaviour"]
    cap_clause = f"fills {cap} slot {pick['cap_position']} of {pick['cap_quota']}"
    if pick["cap_quota"] > pick["cap_target"]:
        cap_clause += f" (above the target of {pick['cap_target']}, because other cap categories had too few eligible stocks)"
    if pick["behaviour_count_before"] == 0:
        balance = f"the first {behaviour} stock in the basket"
    else:
        balance = f"brings {behaviour} stocks in the basket to {pick['behaviour_count_before'] + 1}"
    behaviour_clause = f"{behaviour} on {pick['behaviour_share']:.0%} of its days, {balance}"
    if pick["max_similarity"] is None:
        similarity_clause = "first pick, so no similarity check was needed"
    elif pick["redundant"]:
        similarity_clause = (
            f"similarity {pick['max_similarity']:.2f} with {pick['partner']} is ABOVE the {threshold:.2f} limit, kept only "
            "because every remaining candidate in the open cap categories also exceeded it"
        )
    else:
        similarity_clause = (
            f"highest similarity to already-selected stocks is {pick['max_similarity']:.2f} "
            f"(with {pick['partner']}), within the {threshold:.2f} limit"
        )
    return f"{cap} + {behaviour}; pick {pick['pick_number']}: {cap_clause}; {behaviour_clause}; {similarity_clause}."


# ---------------------------------------------------------------------------
# The selection itself
# ---------------------------------------------------------------------------
def _greedy_select(eligible: pd.DataFrame, similarity: pd.DataFrame, target: dict, quota: dict, basket_size: int, threshold: float) -> list:
    """
    Pick stocks one at a time (see the module notes). `eligible` is indexed by symbol with columns cap_category,
    behavior_class and share. Returns one dict per pick, in pick order, holding the facts the reason is built from.
    """
    cap_of = eligible["cap_category"].to_dict()
    behaviour_of = eligible["behavior_class"].to_dict()
    share_of = eligible["share"].to_dict()
    remaining = set(eligible.index)
    selected: list = []
    cap_count = {cap: 0 for cap in CAP_CATEGORIES}
    behaviour_count = {name: 0 for name in CLASS_NAMES}
    picks = []

    while len(selected) < basket_size:
        open_caps = {cap for cap in CAP_CATEGORIES if cap_count[cap] < quota[cap]}
        candidates = sorted(s for s in remaining if cap_of[s] in open_caps)  # the quotas guarantee there is at least one
        ranked = []
        for symbol in candidates:
            if selected:
                against = similarity.loc[symbol, sorted(selected)]
                max_similarity, partner = float(against.max()), str(against.idxmax())
            else:
                max_similarity, partner = None, None
            redundant = max_similarity is not None and max_similarity > threshold
            key = (
                redundant,  # a. not redundant first
                behaviour_count[behaviour_of[symbol]],  # b. the least represented behaviour class
                round(max_similarity, _SIMILARITY_ROUND) if max_similarity is not None else 0.0,  # c. least similar to the basket
                symbol,  # d. alphabetical
            )
            ranked.append((key, symbol, max_similarity, partner, redundant))
        _, symbol, max_similarity, partner, redundant = min(ranked, key=lambda item: item[0])

        cap, behaviour = cap_of[symbol], behaviour_of[symbol]
        picks.append(
            {
                "symbol": symbol,
                "cap": cap,
                "behaviour": behaviour,
                "behaviour_share": float(share_of[symbol]),
                "pick_number": len(selected) + 1,
                "cap_position": cap_count[cap] + 1,
                "cap_quota": quota[cap],
                "cap_target": target[cap],
                "behaviour_count_before": behaviour_count[behaviour],
                "max_similarity": max_similarity,
                "partner": partner,
                "redundant": redundant,
            }
        )
        selected.append(symbol)
        remaining.discard(symbol)
        cap_count[cap] += 1
        behaviour_count[behaviour] += 1
    return picks


def _supply_notes(target: dict, quota: dict, available: dict) -> list:
    """Plain-English notes for every cap category whose target could not be met."""
    short = [cap for cap in CAP_CATEGORIES if quota[cap] < target[cap]]
    extra = {cap: quota[cap] - target[cap] for cap in CAP_CATEGORIES if quota[cap] > target[cap]}
    notes = []
    for cap in short:
        notes.append(
            f"Target was {_plural(target[cap], f'{cap} stock')}, but only {available[cap]} eligible {cap} "
            f"{'stock was' if available[cap] == 1 else 'stocks were'} available."
        )
    if extra:
        total = sum(extra.values())
        where = ", ".join(f"{cap} ({n})" for cap, n in extra.items()) if len(extra) > 1 else next(iter(extra))
        notes[-1] += f" {_plural(total, 'additional stock')} {'was' if total == 1 else 'were'} selected from {where}."
    return notes


def generate_basket(
    lda_data: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    universe: pd.DataFrame,
    basket_size: int = DEFAULT_BASKET_SIZE,
    capital: float = DEFAULT_CAPITAL,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> BasketResult:
    """
    Build the multi-cap, behaviourally diversified, equal-weight basket (see the module notes for the algorithm).

    Invalid inputs raise a ValueError that says what is wrong; nothing is silently repaired. A cap target that
    cannot be met (too few eligible stocks in a category) does NOT raise: the basket is filled from other
    categories and the shortfall is reported in `notes`. None of the inputs are modified.
    """
    capital = _check_capital(capital)
    basket_size = _check_basket_size(basket_size)
    threshold = _check_threshold(similarity_threshold)
    universe_table = check_universe_table(universe).set_index("symbol")
    behaviour = stock_behaviour(lda_data)
    similarity = _check_similarity_matrix(similarity_matrix)

    notes: list = []
    in_universe = set(universe_table.index)
    no_label = sorted(in_universe - set(behaviour.index))
    no_similarity = sorted((in_universe & set(behaviour.index)) - set(similarity.index))
    not_in_universe = sorted(set(behaviour.index) - in_universe)
    if no_label:
        notes.append(f"Left out (in the universe but no behaviour label): {', '.join(no_label)}.")
    if no_similarity:
        notes.append(f"Left out (not in the similarity matrix): {', '.join(no_similarity)}.")
    if not_in_universe:
        notes.append(f"Ignored (have behaviour labels but are not in the universe): {', '.join(not_in_universe)}.")

    eligible_symbols = sorted(in_universe & set(behaviour.index) & set(similarity.index))
    if basket_size > len(eligible_symbols):
        raise ValueError(f"basket_size ({basket_size}) is larger than the number of eligible stocks ({len(eligible_symbols)}).")
    eligible = universe_table.loc[eligible_symbols].join(behaviour[["behavior_class", "share"]])

    target = cap_targets(basket_size)
    available = {cap: int((eligible["cap_category"] == cap).sum()) for cap in CAP_CATEGORIES}
    quota = plan_cap_quotas(target, available, basket_size)
    notes.extend(_supply_notes(target, quota, available))

    picks = _greedy_select(eligible, similarity, target, quota, basket_size, threshold)

    for pick in picks:
        if pick["redundant"]:
            notes.append(
                f"{pick['symbol']} was selected even though its similarity with {pick['partner']} is "
                f"{pick['max_similarity']:.2f}, above the {threshold:.2f} limit: no less-similar eligible stock was left "
                "in the open cap categories."
            )
    chosen = pd.DataFrame(picks)
    for name in CLASS_NAMES:
        if (chosen["behaviour"] == name).sum() == 0:
            n_available = int((eligible["behavior_class"] == name).sum())
            why = "the eligible universe has none" if n_available == 0 else f"{n_available} eligible, but the cap targets and similarity limit left no room"
            notes.append(f"No {name} stock is in the basket ({why}).")

    weight = 1.0 / basket_size
    rows = pd.DataFrame(
        {
            "symbol": [p["symbol"] for p in picks],
            "cap_category": [p["cap"] for p in picks],
            "behavior_class": [p["behaviour"] for p in picks],
            "weight": weight,
            "allocation": capital * weight,
            "reason": [_reason(p, threshold) for p in picks],
        }
    )
    order = rows["cap_category"].map({cap: i for i, cap in enumerate(CAP_CATEGORIES)})
    basket = rows.assign(_order=order).sort_values(["_order", "symbol"]).drop(columns="_order").reset_index(drop=True)[BASKET_COLUMNS]
    validate_basket(basket, universe, basket_size, capital)
    return BasketResult(
        basket=basket,
        target_allocation=target,
        planned_allocation=quota,
        notes=notes,
        capital=capital,
        basket_size=basket_size,
        similarity_threshold=threshold,
        n_eligible=len(eligible_symbols),
    )


def validate_basket(basket: pd.DataFrame, universe: pd.DataFrame, basket_size: int, capital: float) -> None:
    """Raise a ValueError if a basket breaks any rule: size, duplicates, universe, labels, weights, allocation, reasons."""
    if list(basket.columns) != BASKET_COLUMNS:
        raise ValueError(f"basket columns must be {BASKET_COLUMNS}, got {list(basket.columns)}")
    if len(basket) != basket_size:
        raise ValueError(f"basket has {len(basket)} stocks, expected {basket_size}.")
    repeated = sorted(basket.loc[basket["symbol"].duplicated(), "symbol"].unique())
    if repeated:
        raise ValueError(f"basket contains duplicate stocks: {repeated}")
    table = check_universe_table(universe).set_index("symbol")["cap_category"]
    outside = sorted(set(basket["symbol"]) - set(table.index))
    if outside:
        raise ValueError(f"basket contains stocks that are not in the universe: {outside}")
    if (basket["cap_category"].to_numpy() != table.loc[basket["symbol"]].to_numpy()).any():
        raise ValueError("basket cap_category does not match the universe.")
    invalid = sorted(set(basket["behavior_class"].astype(str)) - set(CLASS_NAMES))
    if invalid:
        raise ValueError(f"basket contains invalid behaviour class(es): {invalid}")
    if not math.isclose(float(basket["weight"].sum()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"basket weights sum to {basket['weight'].sum()!r}, expected 1.")
    if not math.isclose(float(basket["allocation"].sum()), capital, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"basket allocations sum to {basket['allocation'].sum()!r}, expected {capital!r}.")
    if basket["reason"].isna().any() or (basket["reason"].astype(str).str.strip() == "").any():
        raise ValueError("every selected stock needs a reason.")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def pairwise_similarities(symbols, similarity_matrix: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity of every unordered pair of the given stocks, as a table (symbol_a, symbol_b, similarity)."""
    similarity = _check_similarity_matrix(similarity_matrix)
    ordered = sorted(symbols)
    pairs = [(a, b, float(similarity.loc[a, b])) for a, b in combinations(ordered, 2)]
    return pd.DataFrame(pairs, columns=["symbol_a", "symbol_b", "similarity"])


def basket_statistics(basket: pd.DataFrame, similarity_matrix: pd.DataFrame, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> dict:
    """
    Simple diagnostics (no returns, no Sharpe, no drawdown: those belong to Phase 7).

    behaviour_counts / behaviour_percent and cap_counts / cap_percent (every class and category listed, 0 if absent);
    average_similarity and maximum_similarity over all pairs of basket stocks, with the most similar pair;
    pairs_above_threshold. Raises if the basket has fewer than two stocks.
    """
    pairs = pairwise_similarities(basket["symbol"], similarity_matrix)
    if pairs.empty:
        raise ValueError("At least two stocks are needed to compare similarity.")
    n = len(basket)
    behaviour_counts = basket["behavior_class"].astype(str).value_counts().reindex(CLASS_NAMES, fill_value=0).astype(int)
    cap_counts = basket["cap_category"].astype(str).value_counts().reindex(CAP_CATEGORIES, fill_value=0).astype(int)
    top = pairs.loc[pairs["similarity"].idxmax()]
    return {
        "behaviour_counts": behaviour_counts,
        "behaviour_percent": behaviour_counts / n * 100,
        "cap_counts": cap_counts,
        "cap_percent": cap_counts / n * 100,
        "average_similarity": float(pairs["similarity"].mean()),
        "maximum_similarity": float(top["similarity"]),
        "maximum_pair": (str(top["symbol_a"]), str(top["symbol_b"])),
        "pairs_above_threshold": int((pairs["similarity"] > threshold).sum()),
        "n_pairs": len(pairs),
    }


# ---------------------------------------------------------------------------
# Saving and plotting
# ---------------------------------------------------------------------------
def save_basket(basket: pd.DataFrame, path: str | Path) -> Path:
    """Write the basket to CSV (columns: symbol, cap_category, behavior_class, weight, allocation, reason)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    basket[BASKET_COLUMNS].to_csv(path, index=False)
    return path


def plot_basket_allocation(basket: pd.DataFrame, path: str | Path, capital: float, note: str | None = None) -> Path:
    """Horizontal bars of the rupee allocation per stock, grouped Large -> Mid -> Small, coloured by behaviour class."""
    if len(basket) == 0:
        raise ValueError("The basket is empty; nothing to plot.")
    n = len(basket)
    positions = np.arange(n)[::-1]  # first stock on top
    amounts = basket["allocation"].to_numpy()

    fig = Figure(figsize=(9, 1.5 + 0.5 * n), facecolor=_SURFACE)
    ax = fig.subplots()
    _style_axes(ax, grid_axis="x")
    ax.barh(positions, amounts, height=0.55, color=[CLASS_COLOURS[c] for c in basket["behavior_class"].astype(str)])
    ax.set_yticks(positions, basket["symbol"].tolist(), fontsize=10, color=_INK)
    top = float(amounts.max())
    ax.set_xlim(0, top * 1.95)
    for y, row in zip(positions, basket.itertuples(index=False)):
        ax.annotate(
            f"{format_rupees(row.allocation)}  ({row.weight:.1%})  ·  {row.cap_category}  ·  {row.behavior_class}",
            (row.allocation, y), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=9, color=_INK,
        )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: format_rupees(value)))
    ax.set_xlabel("Allocation (equal weights)", color=_INK_2, fontsize=9)
    _title(ax, "Basket allocation by stock",
           f"{format_rupees(capital)} split equally across {n} stocks; bar colour = behavioural class. Not an investment recommendation")
    handles = [Patch(color=CLASS_COLOURS[name], label=name) for name in CLASS_NAMES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=9, labelcolor=_INK_2)
    if note:
        fig.text(0.01, 0.005, note, color=_MUTED, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    return path


# ---------------------------------------------------------------------------
# Report and command-line demo
# ---------------------------------------------------------------------------
def print_basket_report(result: BasketResult, similarity_matrix: pd.DataFrame, source: str = "", eligible_symbols=None) -> dict:
    basket = result.basket
    stats = basket_statistics(basket, similarity_matrix, result.similarity_threshold)
    rule = "-" * 57
    print("=== INTELLIGENT STOCK BASKET ===")
    print()
    if "synthetic" in source.lower():
        print("NOTE: The current dataset is synthetic and is used only to verify that the implementation works.")
    print(DEVELOPMENT_NOTICE)
    print()
    print(f"Capital: {format_rupees(result.capital)}")
    print(f"Basket size: {result.basket_size}")
    print(f"Similarity limit: {result.similarity_threshold:.2f} (candidates above it are penalised)")
    print()
    print("Target cap allocation:")
    for cap in CAP_CATEGORIES:
        print(f"{cap + ':':<11}{result.target_allocation[cap]}")
    print()
    print("Actual basket:")
    print(rule)
    print(f"{'Symbol':<11}{'Cap':<11}{'Behaviour':<12}{'Weight':<9}Allocation")
    print(rule)
    for row in basket.itertuples(index=False):
        print(f"{row.symbol:<11}{row.cap_category:<11}{row.behavior_class:<12}{row.weight * 100:>5.2f}%   {format_rupees(row.allocation)}")
    print(rule)
    print()
    print(f"Total weight: {basket['weight'].sum() * 100:.2f}%")
    print(f"Total allocation: {format_rupees(basket['allocation'].sum())}")
    print()
    print("Behaviour distribution:")
    for name in CLASS_NAMES:
        print(f"{name + ':':<12}{stats['behaviour_counts'][name]}  ({stats['behaviour_percent'][name]:.1f}%)")
    print()
    print("Cap distribution:")
    for cap in CAP_CATEGORIES:
        print(f"{cap + ':':<11}{stats['cap_counts'][cap]}  ({stats['cap_percent'][cap]:.1f}%)")
    print()
    print("Average pairwise similarity:")
    print(f"{stats['average_similarity']:.4f}")
    print()
    print("Maximum pairwise similarity:")
    print(f"{stats['maximum_similarity']:.4f}  ({stats['maximum_pair'][0]} and {stats['maximum_pair'][1]})")
    print(f"Pairs above the {result.similarity_threshold:.2f} limit: {stats['pairs_above_threshold']} of {stats['n_pairs']}")
    if eligible_symbols is not None and len(eligible_symbols) >= 2:
        universe_pairs = pairwise_similarities(eligible_symbols, similarity_matrix)
        print(f"(For context, the average over all {len(eligible_symbols)} eligible stocks is {universe_pairs['similarity'].mean():.4f}.)")
    print()
    print("Selection notes:")
    if result.notes:
        for note in result.notes:
            print(f"- {note}")
    else:
        print("- All cap targets were met and no stock above the similarity limit had to be used.")
    print()
    print("Why each stock was selected:")
    for row in basket.itertuples(index=False):
        print(f"{row.symbol}: {row.reason}")
    print()
    print(METHOD_STATEMENT)
    print(DISCLAIMER)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the exploratory multi-cap, behaviourally diversified basket.")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--basket-size", type=int, default=None, help="number of stocks (default from config, 10)")
    parser.add_argument("--capital", type=float, default=None, help="rupees to allocate (default from config, 100000)")
    parser.add_argument("--similarity-threshold", type=float, default=None, help="redundancy limit, default 0.90")
    args = parser.parse_args()
    try:  # a terminal that cannot show the rupee sign should print '?' rather than crash
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    config = load_config(config_path)
    cfg = config.get("basket", {})
    basket_size = args.basket_size if args.basket_size is not None else cfg.get("basket_size", DEFAULT_BASKET_SIZE)
    capital = args.capital if args.capital is not None else cfg.get("capital", DEFAULT_CAPITAL)
    threshold = args.similarity_threshold if args.similarity_threshold is not None else cfg.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    lda_file = base_dir / config.get("lda", {}).get("output_file", "data/processed/lda_features.csv")
    matrix_file = base_dir / config.get("similarity", {}).get("matrix_file", "data/processed/similarity_matrix.csv")
    output_file = base_dir / cfg.get("output_file", "data/processed/basket.csv")
    plot_file = base_dir / cfg.get("plot_file", "data/processed/plots/basket_allocation.png")

    for path, module in ((lda_file, "src.lda_model"), (matrix_file, "src.similarity")):
        if not path.exists():
            parser.error(f"{path} not found. Run 'python -m {module}' first.")
    lda_data = pd.read_csv(lda_file)
    similarity_matrix = pd.read_csv(matrix_file, index_col=0)
    universe = get_development_universe()

    try:
        result = generate_basket(lda_data, similarity_matrix, universe, basket_size, capital, threshold)
    except ValueError as error:
        parser.error(str(error))

    source = str(config.get("data", {}).get("prices_file", ""))
    eligible = sorted(set(universe["symbol"]) & set(lda_data["symbol"].astype(str)) & set(similarity_matrix.index.astype(str)))
    print_basket_report(result, similarity_matrix, source=source, eligible_symbols=eligible)

    note = "Synthetic development data and synthetic cap classes (not real prices or real caps)" if "synthetic" in source.lower() else None
    csv_path = save_basket(result.basket, output_file)
    plot_path = plot_basket_allocation(result.basket, plot_file, result.capital, note=note)
    print()
    print(f"Basket saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
