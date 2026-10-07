"""
Phase 6a: the stock universe and its market-cap categories.

SYNTHETIC DEVELOPMENT CAP CLASSIFICATIONS - USED ONLY FOR TESTING AND DEMONSTRATION
----------------------------------------------------------------------------------
The synthetic development dataset has 10 symbols, and they are all effectively Large Cap stocks,
so a multi-cap basket cannot be shown with the real classifications. To exercise the basket code
we therefore ASSIGN the 10 symbols to Large / Mid / Small Cap in a deliberately arbitrary way:
alphabetical order, 4 Large, 3 Mid, 3 Small. The assignment says NOTHING about the real market
capitalisation of any of these companies. (For example RELIANCE and TCS are labelled "Small Cap"
here only because they come late in the alphabet.)

When the real universe is connected later (Phase 10), this mapping is replaced by the real
Large / Mid / Small Cap lists. The rest of the code only needs a table with two columns:

    symbol | cap_category

Run it:   python -m src.universe
"""

from __future__ import annotations

import pandas as pd

DEVELOPMENT_NOTICE = "Synthetic development cap classifications used only for testing and demonstration."

# Ordered from largest to smallest. This order is used everywhere (tables, targets, plots).
CAP_CATEGORIES = ["Large Cap", "Mid Cap", "Small Cap"]
UNIVERSE_COLUMNS = ["symbol", "cap_category"]

# Alphabetical split, on purpose: nothing about it should look like a real classification.
_DEVELOPMENT_ASSIGNMENT = {
    "AXISBANK": "Large Cap",
    "HDFCBANK": "Large Cap",
    "ICICIBANK": "Large Cap",
    "INFY": "Large Cap",
    "ITC": "Mid Cap",
    "LT": "Mid Cap",
    "MARUTI": "Mid Cap",
    "RELIANCE": "Small Cap",
    "SBIN": "Small Cap",
    "TCS": "Small Cap",
}


def check_universe_table(universe: pd.DataFrame) -> pd.DataFrame:
    """
    Check a universe table (symbol, cap_category) and return a clean copy, ordered Large, Mid, Small, then by symbol.

    Nothing is silently repaired: missing columns, an empty table, missing or blank values, repeated symbols and
    unknown cap categories all raise a ValueError. The input table is not modified.
    """
    if not isinstance(universe, pd.DataFrame):
        raise ValueError("universe must be a pandas DataFrame with columns: symbol, cap_category.")
    missing = [c for c in UNIVERSE_COLUMNS if c not in universe.columns]
    if missing:
        raise ValueError(f"universe is missing column(s): {missing}")
    if len(universe) == 0:
        raise ValueError("universe is empty.")
    table = universe[UNIVERSE_COLUMNS].copy()
    if table.isna().any().any():
        raise ValueError("universe has missing symbol or cap_category values.")
    table = table.astype(str)
    if (table["symbol"].str.strip() == "").any() or (table["symbol"] != table["symbol"].str.strip()).any():
        raise ValueError("universe has blank symbols or symbols with leading/trailing spaces.")
    repeated = sorted(table.loc[table["symbol"].duplicated(), "symbol"].unique())
    if repeated:
        raise ValueError(f"universe has duplicate symbols: {repeated}")
    unknown = sorted(set(table["cap_category"]) - set(CAP_CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown cap_category value(s): {unknown}. Allowed: {CAP_CATEGORIES}")
    order = table["cap_category"].map({c: i for i, c in enumerate(CAP_CATEGORIES)})
    return table.assign(_order=order).sort_values(["_order", "symbol"]).drop(columns="_order").reset_index(drop=True)


def get_development_universe() -> pd.DataFrame:
    """
    The development universe: one row per symbol with columns symbol, cap_category.

    SYNTHETIC cap classifications used only for testing and demonstration (see the module notes).
    Returns a new table each time, so changing it never changes the built-in mapping.
    """
    table = pd.DataFrame({"symbol": list(_DEVELOPMENT_ASSIGNMENT), "cap_category": list(_DEVELOPMENT_ASSIGNMENT.values())})
    return check_universe_table(table)


def get_cap_category(symbol: str, universe: pd.DataFrame | None = None) -> str:
    """The cap category of one symbol (from the development universe unless another table is given)."""
    table = get_development_universe() if universe is None else check_universe_table(universe)
    match = table.loc[table["symbol"] == symbol, "cap_category"]
    if match.empty:
        raise ValueError(f"Unknown symbol {symbol!r}. Symbols in the universe: {', '.join(table['symbol'])}")
    return match.iloc[0]


def validate_universe(symbols, universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Check that every symbol in `symbols` exists in the universe and appears only once.

    Returns the matching universe rows (symbol, cap_category) in the order the symbols were given.
    Unknown symbols and repeated symbols raise a ValueError that names them.
    """
    table = get_development_universe() if universe is None else check_universe_table(universe)
    if isinstance(symbols, str):
        raise ValueError("symbols must be a list of symbols, not a single string.")
    symbols = list(symbols)
    if not symbols:
        raise ValueError("symbols is empty.")
    repeated = sorted({s for s in symbols if symbols.count(s) > 1})
    if repeated:
        raise ValueError(f"Duplicate symbols requested: {repeated}")
    unknown = [s for s in symbols if s not in set(table["symbol"])]
    if unknown:
        raise ValueError(f"Unknown symbol(s) {unknown}. Symbols in the universe: {', '.join(table['symbol'])}")
    return table.set_index("symbol").loc[symbols].reset_index()


def main() -> None:
    universe = get_development_universe()
    print("=== DEVELOPMENT UNIVERSE ===")
    print()
    print(DEVELOPMENT_NOTICE)
    print()
    print(universe.to_string(index=False))
    print()
    for cap, count in universe["cap_category"].value_counts().reindex(CAP_CATEGORIES).items():
        print(f"{cap}: {count}")


if __name__ == "__main__":
    main()
