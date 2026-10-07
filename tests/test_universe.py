"""
Tests for Phase 6a (src/universe.py): the synthetic development universe.

Run:  pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.sample_data import DEV_STOCKS
from src.universe import (
    CAP_CATEGORIES,
    DEVELOPMENT_NOTICE,
    UNIVERSE_COLUMNS,
    check_universe_table,
    get_cap_category,
    get_development_universe,
    main,
    validate_universe,
)


# ======================= test 1: it exists =======================
def test_development_universe_exists_with_the_right_columns():
    universe = get_development_universe()
    assert isinstance(universe, pd.DataFrame)
    assert list(universe.columns) == ["symbol", "cap_category"] == UNIVERSE_COLUMNS
    assert len(universe) == 10


# ======================= test 2: all 10 development symbols =======================
def test_every_development_symbol_is_represented_exactly_once():
    universe = get_development_universe()
    assert set(universe["symbol"]) == set(DEV_STOCKS)
    assert universe["symbol"].is_unique and len(universe) == len(DEV_STOCKS)


# ======================= tests 3 and 4: categories and distribution =======================
def test_all_three_cap_categories_exist():
    assert CAP_CATEGORIES == ["Large Cap", "Mid Cap", "Small Cap"]
    assert set(get_development_universe()["cap_category"]) == set(CAP_CATEGORIES)


def test_distribution_is_4_large_3_mid_3_small():
    counts = get_development_universe()["cap_category"].value_counts()
    assert counts.to_dict() == {"Large Cap": 4, "Mid Cap": 3, "Small Cap": 3}


def test_the_table_order_is_deterministic_large_then_mid_then_small():
    universe = get_development_universe()
    pd.testing.assert_frame_equal(universe, get_development_universe())
    assert universe["cap_category"].tolist() == ["Large Cap"] * 4 + ["Mid Cap"] * 3 + ["Small Cap"] * 3
    for _, group in universe.groupby("cap_category"):
        assert group["symbol"].is_monotonic_increasing


def test_the_assignment_is_clearly_labelled_as_synthetic():
    assert DEVELOPMENT_NOTICE == "Synthetic development cap classifications used only for testing and demonstration."
    import src.universe as module

    assert DEVELOPMENT_NOTICE in module.__doc__ or "SYNTHETIC DEVELOPMENT CAP CLASSIFICATIONS" in module.__doc__


def test_changing_the_returned_table_does_not_change_the_built_in_mapping():
    universe = get_development_universe()
    universe.loc[0, "cap_category"] = "Small Cap"
    universe.loc[len(universe)] = ["NEWSTOCK", "Mid Cap"]
    again = get_development_universe()
    assert len(again) == 10 and (again["cap_category"] == "Small Cap").sum() == 3


# ======================= test 5: unknown symbols =======================
def test_unknown_symbol_gives_a_clear_error():
    with pytest.raises(ValueError, match="Unknown symbol 'NOPE'.*Symbols in the universe: AXISBANK"):
        get_cap_category("NOPE")
    with pytest.raises(ValueError, match="Unknown symbol"):
        get_cap_category("tcs")  # exact match only
    with pytest.raises(ValueError, match=r"Unknown symbol\(s\) \['NOPE', 'ZZZ'\]"):
        validate_universe(["AXISBANK", "NOPE", "ZZZ"])


def test_known_symbols_return_their_category():
    universe = get_development_universe()
    for row in universe.itertuples(index=False):
        assert get_cap_category(row.symbol) == row.cap_category
    assert get_cap_category("AXISBANK") == "Large Cap" and get_cap_category("TCS") == "Small Cap"


def test_a_custom_universe_table_can_be_used_instead():
    custom = pd.DataFrame({"symbol": ["A", "B"], "cap_category": ["Small Cap", "Large Cap"]})
    assert get_cap_category("A", custom) == "Small Cap"
    assert validate_universe(["B", "A"], custom)["symbol"].tolist() == ["B", "A"]
    with pytest.raises(ValueError, match="Unknown symbol 'AXISBANK'"):
        get_cap_category("AXISBANK", custom)


# ======================= test 6: duplicates =======================
def test_duplicate_requested_symbols_are_rejected():
    with pytest.raises(ValueError, match=r"Duplicate symbols requested: \['TCS'\]"):
        validate_universe(["TCS", "INFY", "TCS"])


def test_a_universe_table_with_duplicate_symbols_is_rejected():
    table = pd.DataFrame({"symbol": ["A", "B", "A"], "cap_category": ["Large Cap", "Mid Cap", "Small Cap"]})
    with pytest.raises(ValueError, match=r"duplicate symbols: \['A'\]"):
        check_universe_table(table)
    with pytest.raises(ValueError, match="duplicate symbols"):
        get_cap_category("A", table)


# ======================= validate_universe =======================
def test_validate_universe_returns_the_requested_rows_in_order():
    result = validate_universe(["TCS", "AXISBANK", "ITC"])
    assert result["symbol"].tolist() == ["TCS", "AXISBANK", "ITC"]
    assert result["cap_category"].tolist() == ["Small Cap", "Large Cap", "Mid Cap"]
    assert list(result.columns) == ["symbol", "cap_category"]
    assert len(validate_universe(list(DEV_STOCKS))) == 10


@pytest.mark.parametrize("bad", ["TCS", [], ()])
def test_validate_universe_rejects_a_bare_string_or_nothing(bad):
    with pytest.raises(ValueError):
        validate_universe(bad)


# ======================= check_universe_table =======================
def test_the_table_checker_returns_a_sorted_copy_and_leaves_the_input_alone():
    table = pd.DataFrame({"symbol": ["Z", "A", "M"], "cap_category": ["Small Cap", "Mid Cap", "Large Cap"], "extra": [1, 2, 3]})
    before = table.copy(deep=True)
    clean = check_universe_table(table)
    assert clean["symbol"].tolist() == ["M", "A", "Z"] and list(clean.columns) == ["symbol", "cap_category"]
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize(
    "table, message",
    [
        (pd.DataFrame({"symbol": ["A"]}), "missing column"),
        (pd.DataFrame({"symbol": [], "cap_category": []}), "empty"),
        (pd.DataFrame({"symbol": ["A", None], "cap_category": ["Large Cap", "Mid Cap"]}), "missing"),
        (pd.DataFrame({"symbol": ["A", "B"], "cap_category": ["Large Cap", np.nan]}), "missing"),
        (pd.DataFrame({"symbol": ["A", " "], "cap_category": ["Large Cap", "Mid Cap"]}), "blank"),
        (pd.DataFrame({"symbol": ["A", "B "], "cap_category": ["Large Cap", "Mid Cap"]}), "blank"),
        (pd.DataFrame({"symbol": ["A", "B"], "cap_category": ["Large Cap", "Medium Cap"]}), "Unknown cap_category.*Medium Cap"),
        (pd.DataFrame({"symbol": ["A", "B"], "cap_category": ["large cap", "Mid Cap"]}), "Unknown cap_category"),
    ],
)
def test_bad_universe_tables_are_rejected_not_repaired(table, message):
    with pytest.raises(ValueError, match=message):
        check_universe_table(table)


def test_a_non_table_is_rejected():
    with pytest.raises(ValueError, match="DataFrame"):
        check_universe_table([("A", "Large Cap")])


# ======================= command line =======================
def test_command_line_prints_the_notice_and_the_distribution(capsys):
    main()
    text = capsys.readouterr().out
    assert DEVELOPMENT_NOTICE in text
    for expected in ["AXISBANK", "Large Cap: 4", "Mid Cap: 3", "Small Cap: 3"]:
        assert expected in text
