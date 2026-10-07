"""
Tests for Phase 6b (src/basket.py): the rule-based multi-cap, behaviourally diversified basket.

Most tests use small designed "worlds": a handful of stocks with known cap categories, known behaviour
classes and a similarity matrix built from angles (cos 0 degrees = 1, cos 90 = 0, cos 180 = -1), so the
right answer can be worked out on paper. A few tests run the real Phase 2 -> 5 pipeline on the synthetic
development data. They check that the code is correct; they say nothing about real markets.

Run:  pytest -q
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.basket import (
    BASKET_COLUMNS,
    DISCLAIMER,
    METHOD_STATEMENT,
    BasketResult,
    basket_statistics,
    cap_targets,
    format_rupees,
    generate_basket,
    main,
    pairwise_similarities,
    plan_cap_quotas,
    plot_basket_allocation,
    print_basket_report,
    save_basket,
    stock_behaviour,
    validate_basket,
)
from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import build_features, clean_feature_data
from src.labels import CLASS_NAMES, create_behavior_labels, fit_label_rules
from src.lda_model import fit_lda, transform_lda
from src.pca_model import fit_pca, transform_pca
from src.similarity import calculate_similarity, create_stock_profiles
from src.universe import CAP_CATEGORIES, DEVELOPMENT_NOTICE, get_development_universe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ======================= helpers: small designed worlds =======================
def make_world(spec, angles=None, rows_per_stock=5):
    """
    Inputs for generate_basket from spec = [(symbol, cap_category, behaviour_class), ...].
    angles: one angle in degrees per stock; similarity(i, j) = cos(angle_i - angle_j). None = all unrelated (0).
    Returns (lda_data, similarity_matrix, universe).
    """
    symbols = [s[0] for s in spec]
    universe = pd.DataFrame({"symbol": symbols, "cap_category": [s[1] for s in spec]})
    lda = pd.DataFrame(
        [
            {"date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), "symbol": s, "behavior_class": b, "LD1": float(i), "LD2": -float(i)}
            for s, _, b in spec
            for i in range(rows_per_stock)
        ]
    )
    if angles is None:
        matrix = np.eye(len(symbols))
    else:
        radians = np.radians(angles)
        matrix = np.cos(radians[:, None] - radians[None, :])
    return lda, pd.DataFrame(matrix, index=symbols, columns=symbols), universe


def spread_spec(n_large, n_mid, n_small, classes=CLASS_NAMES):
    """Stocks L01.., M01.., S01.. with behaviour classes cycling Defensive, Balanced, Aggressive."""
    names = [(f"L{i + 1:02d}", "Large Cap") for i in range(n_large)]
    names += [(f"M{i + 1:02d}", "Mid Cap") for i in range(n_mid)]
    names += [(f"S{i + 1:02d}", "Small Cap") for i in range(n_small)]
    return [(symbol, cap, classes[i % len(classes)]) for i, (symbol, cap) in enumerate(names)]


def basket_of(spec, n, capital=100_000, threshold=0.90, angles=None):
    lda, sim, universe = make_world(spec, angles)
    return generate_basket(lda, sim, universe, n, capital, threshold)


def symbols_of(result):
    return set(result.basket["symbol"])


def cap_counts(result):
    return result.basket["cap_category"].value_counts().reindex(CAP_CATEGORIES, fill_value=0).to_dict()


@pytest.fixture
def world():
    return make_world(spread_spec(5, 5, 5))


# The "clone" world: MA is almost the same stock as LA (1 degree apart, similarity 0.99985) but has a
# behaviour class the basket does not have yet, so only the similarity rule can keep it out.
CLONE_SPEC = [("LA", "Large Cap", "Defensive"), ("MA", "Mid Cap", "Balanced"), ("MB", "Mid Cap", "Defensive"), ("SA", "Small Cap", "Aggressive")]
CLONE_ANGLES = [0, 1, 120, 240]


# ======================= cap targets =======================
@pytest.mark.parametrize("n, expected", [(3, (1, 1, 1)), (4, (2, 1, 1)), (6, (2, 2, 2)), (10, (4, 3, 3)), (15, (6, 5, 4)), (20, (8, 6, 6))])
def test_cap_targets_follow_the_documented_rounding(n, expected):
    assert tuple(cap_targets(n).values()) == expected
    assert list(cap_targets(n)) == CAP_CATEGORIES


@pytest.mark.parametrize("n", range(3, 61))
def test_cap_targets_always_sum_to_the_basket_size_and_give_every_cap_a_place(n):
    targets = cap_targets(n)
    assert sum(targets.values()) == n
    assert min(targets.values()) >= 1
    assert targets["Large Cap"] >= targets["Mid Cap"] >= targets["Small Cap"]
    assert max(targets.values()) - min(targets.values()) <= max(2, round(0.12 * n))  # stays roughly proportional


def test_cap_targets_reject_invalid_sizes():
    with pytest.raises(ValueError, match="at least 3"):
        cap_targets(2)


@pytest.mark.parametrize(
    "target, available, n, expected",
    [
        ({"Large Cap": 4, "Mid Cap": 3, "Small Cap": 3}, {"Large Cap": 9, "Mid Cap": 9, "Small Cap": 9}, 10, (4, 3, 3)),
        ({"Large Cap": 4, "Mid Cap": 3, "Small Cap": 3}, {"Large Cap": 4, "Mid Cap": 6, "Small Cap": 2}, 10, (4, 4, 2)),  # Mid has the most spare stocks
        ({"Large Cap": 2, "Mid Cap": 2, "Small Cap": 2}, {"Large Cap": 4, "Mid Cap": 4, "Small Cap": 1}, 6, (3, 2, 1)),  # tie: the larger cap first
        ({"Large Cap": 2, "Mid Cap": 2, "Small Cap": 2}, {"Large Cap": 4, "Mid Cap": 4, "Small Cap": 0}, 6, (3, 3, 0)),
        ({"Large Cap": 2, "Mid Cap": 2, "Small Cap": 2}, {"Large Cap": 0, "Mid Cap": 0, "Small Cap": 6}, 6, (0, 0, 6)),
    ],
)
def test_quotas_move_missing_slots_to_the_category_with_the_most_spare_stocks(target, available, n, expected):
    assert tuple(plan_cap_quotas(target, available, n).values()) == expected


def test_quotas_refuse_an_impossible_basket():
    with pytest.raises(ValueError, match="Not enough eligible"):
        plan_cap_quotas(cap_targets(6), {"Large Cap": 1, "Mid Cap": 1, "Small Cap": 1}, 6)


# ======================= test 1: correct basket size =======================
@pytest.mark.parametrize("n", [3, 4, 6, 10, 15])
def test_basket_has_the_requested_size(world, n):
    result = generate_basket(*world, basket_size=n)
    assert len(result.basket) == n == result.basket_size
    assert isinstance(result, BasketResult)


# ======================= test 2: no duplicates =======================
@pytest.mark.parametrize("n", [3, 7, 15])
def test_no_stock_appears_twice(world, n):
    assert generate_basket(*world, basket_size=n).basket["symbol"].is_unique


def test_selecting_every_eligible_stock_selects_each_exactly_once(world):
    result = generate_basket(*world, basket_size=15)
    assert sorted(result.basket["symbol"]) == sorted(world[2]["symbol"])


# ======================= tests 3, 4, 5: weights and allocation =======================
@pytest.mark.parametrize("n", [3, 4, 5, 7, 8, 10])
def test_weights_sum_to_one(world, n):
    assert generate_basket(*world, basket_size=n).basket["weight"].sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("capital", [100_000, 50_000, 12_345.67, 1, 1e9])
def test_allocations_sum_to_the_capital(world, capital):
    for n in (6, 7, 10):
        result = generate_basket(*world, basket_size=n, capital=capital)
        assert result.basket["allocation"].sum() == pytest.approx(capital, rel=1e-12)
        assert result.capital == capital


@pytest.mark.parametrize("n, weight", [(10, 0.10), (5, 0.20), (8, 0.125), (3, 1 / 3)])
def test_equal_weighting_is_exactly_one_over_the_basket_size(world, n, weight):
    basket = generate_basket(*world, basket_size=n, capital=80_000).basket
    assert basket["weight"].to_numpy() == pytest.approx(weight)
    assert basket["weight"].to_numpy() == pytest.approx(1 / n)
    assert basket["allocation"].to_numpy() == pytest.approx(80_000 * weight)
    assert basket["weight"].nunique() == 1 and basket["allocation"].nunique() == 1


# ======================= tests 6, 7: capital and size checks =======================
@pytest.mark.parametrize("bad", [0, -1, -100_000, float("nan"), float("inf"), True, "100000", None])
def test_capital_must_be_a_positive_number(world, bad):
    with pytest.raises(ValueError, match="capital must be a positive number"):
        generate_basket(*world, basket_size=6, capital=bad)


@pytest.mark.parametrize("bad", [2, 1, 0, -5, 2.5, 6.0, True, "6", None])
def test_basket_size_must_be_a_whole_number_of_at_least_three(world, bad):
    with pytest.raises(ValueError, match="basket_size must be"):
        generate_basket(*world, basket_size=bad)


def test_basket_size_error_messages_say_what_is_wrong(world):
    with pytest.raises(ValueError, match="at least 3, got 2"):
        generate_basket(*world, basket_size=2)
    with pytest.raises(ValueError, match="whole number"):
        generate_basket(*world, basket_size=4.5)


# ======================= test 8: not more stocks than exist =======================
def test_cannot_ask_for_more_stocks_than_are_eligible():
    lda, sim, universe = make_world(spread_spec(2, 2, 2))
    with pytest.raises(ValueError, match=r"basket_size \(7\) is larger than the number of eligible stocks \(6\)"):
        generate_basket(lda, sim, universe, basket_size=7)
    assert len(generate_basket(lda, sim, universe, basket_size=6).basket) == 6  # exactly all of them is fine


def test_only_eligible_stocks_count_toward_what_is_available():
    lda, sim, universe = make_world(spread_spec(3, 3, 3))
    lda = lda[~lda["symbol"].isin(["L01", "M01", "S01"])]  # three universe stocks have no behaviour label
    with pytest.raises(ValueError, match=r"larger than the number of eligible stocks \(6\)"):
        generate_basket(lda, sim, universe, basket_size=7)
    result = generate_basket(lda, sim, universe, basket_size=6)
    assert not symbols_of(result) & {"L01", "M01", "S01"}
    assert any("no behaviour label" in note and "L01, M01, S01" in note for note in result.notes)


def test_stocks_missing_from_the_similarity_matrix_are_left_out_and_reported():
    lda, sim, universe = make_world(spread_spec(3, 3, 3))
    sim = sim.drop(index=["S03"], columns=["S03"])
    result = generate_basket(lda, sim, universe, basket_size=8)
    assert "S03" not in symbols_of(result)
    assert any("not in the similarity matrix" in note and "S03" in note for note in result.notes)
    with pytest.raises(ValueError, match=r"eligible stocks \(8\)"):
        generate_basket(lda, sim, universe, basket_size=9)


# ======================= test 9: universe membership =======================
def test_every_selected_stock_exists_in_the_universe_and_with_its_universe_category():
    lda, sim, universe = make_world(spread_spec(4, 4, 4))
    extra = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")] * 3, "symbol": ["ZZZ"] * 3, "behavior_class": ["Defensive"] * 3, "LD1": 0.0, "LD2": 0.0})
    sim.loc["ZZZ", :] = 0.0
    sim["ZZZ"] = 0.0
    sim.loc["ZZZ", "ZZZ"] = 1.0
    result = generate_basket(pd.concat([lda, extra]), sim, universe, basket_size=12)
    assert "ZZZ" not in symbols_of(result)
    assert set(result.basket["symbol"]) <= set(universe["symbol"])
    assert any("not in the universe" in note and "ZZZ" in note for note in result.notes)
    expected = universe.set_index("symbol")["cap_category"]
    for row in result.basket.itertuples(index=False):
        assert row.cap_category == expected[row.symbol]


def test_the_universe_table_is_validated(world):
    lda, sim, universe = world
    with pytest.raises(ValueError, match="duplicate symbols"):
        generate_basket(lda, sim, pd.concat([universe, universe.head(1)]), basket_size=6)
    with pytest.raises(ValueError, match="Unknown cap_category"):
        generate_basket(lda, sim, universe.assign(cap_category="Micro Cap"), basket_size=6)
    with pytest.raises(ValueError, match="missing column"):
        generate_basket(lda, sim, universe[["symbol"]], basket_size=6)


# ======================= test 10: valid behaviour classes =======================
def test_every_selected_stock_has_a_valid_behaviour_class(world):
    basket = generate_basket(*world, basket_size=10).basket
    assert set(basket["behavior_class"]) <= set(CLASS_NAMES)
    assert basket["behavior_class"].notna().all()


def test_invalid_or_missing_labels_are_rejected_not_skipped(world):
    lda, sim, universe = world
    with pytest.raises(ValueError, match=r"invalid behaviour class.*Neutral"):
        generate_basket(lda.assign(behavior_class="Neutral"), sim, universe, basket_size=6)
    broken = lda.copy()
    broken.loc[broken.index[3], "behavior_class"] = np.nan
    with pytest.raises(ValueError, match="missing symbol or behavior_class"):
        generate_basket(broken, sim, universe, basket_size=6)
    with pytest.raises(ValueError, match="missing column"):
        generate_basket(lda.drop(columns=["behavior_class"]), sim, universe, basket_size=6)
    with pytest.raises(ValueError, match="empty"):
        generate_basket(lda.head(0), sim, universe, basket_size=6)


def test_a_stocks_behaviour_is_its_most_frequent_class():
    rows = (
        [("X", "Defensive")] * 2 + [("X", "Balanced")]
        + [("Y", "Defensive"), ("Y", "Balanced")]  # a tie
        + [("Z", "Aggressive")] * 2 + [("Z", "Balanced")] * 3
    )
    table = stock_behaviour(pd.DataFrame(rows, columns=["symbol", "behavior_class"]))
    assert table["behavior_class"].to_dict() == {"X": "Defensive", "Y": "Defensive", "Z": "Balanced"}  # ties go to the first class in order
    assert table["share"].to_dict() == pytest.approx({"X": 2 / 3, "Y": 1 / 2, "Z": 3 / 5})
    assert table["n_observations"].to_dict() == {"X": 3, "Y": 2, "Z": 5}


def test_an_lda_table_with_repeated_index_labels_still_works(world):
    """Two tables stacked with pd.concat repeat their index labels; the stock's class must not depend on the index."""
    lda, sim, universe = world
    stacked = pd.concat([lda, lda.assign(date=lda["date"] + pd.Timedelta(days=30))])
    assert not stacked.index.is_unique
    pd.testing.assert_frame_equal(stock_behaviour(stacked)[["behavior_class"]], stock_behaviour(lda)[["behavior_class"]])
    assert len(generate_basket(stacked, sim, universe, 6).basket) == 6


def test_categorical_labels_from_the_lda_output_work(world):
    lda, sim, universe = world
    lda = lda.assign(behavior_class=pd.Categorical(lda["behavior_class"], categories=CLASS_NAMES, ordered=True))
    assert len(generate_basket(lda, sim, universe, basket_size=6).basket) == 6


def test_the_basket_depends_only_on_the_labels_not_on_ld_scores_or_dates(world):
    lda, sim, universe = world
    changed = lda.assign(LD1=lda["LD1"] * -50 + 7, LD2=0.0, date=pd.Timestamp("2030-01-01"))
    pd.testing.assert_frame_equal(generate_basket(lda, sim, universe, 7).basket, generate_basket(changed, sim, universe, 7).basket)
    labels_only = lda[["symbol", "behavior_class"]]
    pd.testing.assert_frame_equal(generate_basket(lda, sim, universe, 7).basket, generate_basket(labels_only, sim, universe, 7).basket)


def test_the_generator_takes_no_price_return_or_feature_inputs():
    """Leakage rule: selection cannot see the future because it is never given prices or returns."""
    assert list(inspect.signature(generate_basket).parameters) == [
        "lda_data", "similarity_matrix", "universe", "basket_size", "capital", "similarity_threshold",
    ]


# ======================= test 11: cap diversification =======================
@pytest.mark.parametrize("n, expected", [(10, (4, 3, 3)), (6, (2, 2, 2)), (15, (6, 5, 4)), (3, (1, 1, 1)), (8, (3, 3, 2))])
def test_cap_categories_follow_the_target_when_enough_stocks_exist(n, expected):
    result = basket_of(spread_spec(8, 8, 8), n)
    assert tuple(cap_counts(result).values()) == expected
    assert result.target_allocation == result.planned_allocation == dict(zip(CAP_CATEGORIES, expected))
    assert not any("Target was" in note for note in result.notes)


@pytest.mark.parametrize("n", range(3, 16))
def test_every_cap_category_is_represented_at_every_size(n):
    assert min(cap_counts(basket_of(spread_spec(8, 8, 8), n)).values()) >= 1


# ======================= test 12: behavioural diversification =======================
def test_all_three_behaviour_classes_are_included_when_available():
    result = basket_of(spread_spec(3, 3, 3), 3)
    assert set(result.basket["behavior_class"]) == set(CLASS_NAMES)
    assert result.basket["behavior_class"].nunique() == 3


@pytest.mark.parametrize("n", [3, 4, 5, 6, 9])
def test_the_basket_is_never_dominated_by_one_class_when_alternatives_exist(n):
    counts = basket_of(spread_spec(6, 6, 6), n).basket["behavior_class"].value_counts()
    assert len(counts) == 3
    assert counts.max() - counts.min() <= 1  # classes are as even as the quotas allow


def test_a_scarce_class_is_picked_early_instead_of_being_missed():
    spec = [(s, c, "Defensive") for s, c, _ in spread_spec(3, 3, 3)]
    spec[-1] = (spec[-1][0], spec[-1][1], "Aggressive")  # the only Aggressive stock
    result = basket_of(spec, 4)
    assert "Aggressive" in set(result.basket["behavior_class"])
    assert result.basket.loc[result.basket["behavior_class"] == "Aggressive", "symbol"].tolist() == ["S03"]


def test_a_universe_with_one_class_still_gives_a_basket_and_says_what_is_missing():
    result = basket_of([(s, c, "Defensive") for s, c, _ in spread_spec(3, 3, 3)], 6)
    assert set(result.basket["behavior_class"]) == {"Defensive"} and len(result.basket) == 6
    assert "No Balanced stock is in the basket (the eligible universe has none)." in result.notes
    assert "No Aggressive stock is in the basket (the eligible universe has none)." in result.notes


# ======================= tests 13, 14: similarity-based redundancy control =======================
def test_a_near_clone_is_avoided_when_a_less_similar_alternative_exists():
    result = basket_of(CLONE_SPEC, 3, angles=CLONE_ANGLES)
    assert symbols_of(result) == {"LA", "MB", "SA"}  # MA (similarity 0.99985 with LA) is skipped
    assert "MA" not in symbols_of(result)
    lda, sim, universe = make_world(CLONE_SPEC, CLONE_ANGLES)
    assert basket_statistics(result.basket, sim)["maximum_similarity"] < 0.90
    assert not any("ABOVE" in reason for reason in result.basket["reason"])


def test_without_the_similarity_rule_the_same_clone_is_chosen():
    """The contrast that proves the rule matters: MA's class is under-represented, so only the threshold keeps it out."""
    result = basket_of(CLONE_SPEC, 3, threshold=1.0, angles=CLONE_ANGLES)
    assert "MA" in symbols_of(result) and "MB" not in symbols_of(result)
    lda, sim, universe = make_world(CLONE_SPEC, CLONE_ANGLES)
    assert basket_statistics(result.basket, sim)["maximum_similarity"] > 0.99


def test_among_otherwise_equal_candidates_the_least_similar_to_the_basket_wins():
    spec = [("A", "Large Cap", "Defensive"), ("B1", "Mid Cap", "Defensive"), ("B2", "Mid Cap", "Defensive"), ("C", "Small Cap", "Defensive")]
    result = basket_of(spec, 3, angles=[0, 30, 90, 180])  # B1 is closer to A (0.87) than B2 (0.00); none exceeds the 0.90 limit
    assert symbols_of(result) == {"A", "B2", "C"}
    pick2 = result.basket.set_index("symbol").loc["C", "reason"]
    assert "pick 2:" in pick2 and "-1.00 (with A)" in pick2  # C (-1.00) beat B1 (0.87) and B2 (0.00)


def test_the_similarity_threshold_is_configurable():
    cos_one_degree = np.cos(np.radians(1))  # 0.99985
    assert 0.999 < cos_one_degree < 0.9999
    blocked = basket_of(CLONE_SPEC, 3, threshold=0.999, angles=CLONE_ANGLES)
    allowed = basket_of(CLONE_SPEC, 3, threshold=0.9999, angles=CLONE_ANGLES)
    assert "MA" not in symbols_of(blocked) and "MA" in symbols_of(allowed)
    assert blocked.similarity_threshold == 0.999 and allowed.similarity_threshold == 0.9999


def test_the_default_threshold_is_0_90_and_the_limit_is_strictly_above():
    assert inspect.signature(generate_basket).parameters["similarity_threshold"].default == 0.90
    exactly_at_limit = [("A", "Large Cap", "Defensive"), ("B", "Mid Cap", "Balanced"), ("C", "Small Cap", "Aggressive")]
    lda, sim, universe = make_world(exactly_at_limit)
    sim.loc["A", "B"] = sim.loc["B", "A"] = 0.90
    result = generate_basket(lda, sim, universe, 3)
    assert not any("ABOVE" in reason for reason in result.basket["reason"])  # 0.90 is allowed, only > 0.90 is penalised
    sim.loc["A", "B"] = sim.loc["B", "A"] = 0.9001
    assert any("ABOVE" in reason for reason in generate_basket(lda, sim, universe, 3).basket["reason"])


@pytest.mark.parametrize("bad", [1.01, -1.5, float("nan"), "0.9", None, True])
def test_the_threshold_must_be_a_number_between_minus_one_and_one(world, bad):
    with pytest.raises(ValueError, match="similarity_threshold"):
        generate_basket(*world, basket_size=6, similarity_threshold=bad)


def test_unavoidable_redundancy_is_used_as_a_last_resort_and_reported():
    spec = [("LA", "Large Cap", "Defensive"), ("LB", "Large Cap", "Defensive"), ("MA", "Mid Cap", "Balanced"), ("SA", "Small Cap", "Aggressive")]
    result = basket_of(spec, 4, angles=[0, 2, 120, 240])  # all four stocks must be picked, and LA/LB are near-clones
    assert symbols_of(result) == {"LA", "LB", "MA", "SA"}
    assert any("LB was selected even though its similarity with LA is 1.00" in note for note in result.notes)
    reason = result.basket.set_index("symbol").loc["LB", "reason"]
    assert "ABOVE the 0.90 limit" in reason and "every remaining candidate" in reason


def test_the_similarity_matrix_may_arrive_in_any_order_or_loaded_from_csv(tmp_path):
    lda, sim, universe = make_world(CLONE_SPEC, CLONE_ANGLES)
    expected = generate_basket(lda, sim, universe, 3).basket
    shuffled = sim.loc[["SA", "MB", "LA", "MA"], ["MB", "SA", "MA", "LA"]]
    pd.testing.assert_frame_equal(generate_basket(lda, shuffled, universe, 3).basket, expected)
    sim.to_csv(tmp_path / "m.csv")
    pd.testing.assert_frame_equal(generate_basket(lda, pd.read_csv(tmp_path / "m.csv", index_col=0), universe, 3).basket, expected)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda s: s.iloc[:-1, :], "square"),
        (lambda s: s.rename(index={"L01": "OTHER"}), "same symbols"),
        (lambda s: s.mask(s.index.to_series().eq("L02").to_numpy()[:, None] & s.columns.to_series().eq("M01").to_numpy()[None, :], np.nan), "missing or infinite"),
        (lambda s: s * 1.5, "outside"),
        (lambda s: s.mask(s.index.to_series().eq("L01").to_numpy()[:, None] & s.columns.to_series().eq("M01").to_numpy()[None, :], 0.3), "symmetric"),
    ],
)
def test_a_bad_similarity_matrix_is_rejected(mutate, message):
    lda, sim, universe = make_world(spread_spec(2, 2, 2))
    with pytest.raises(ValueError, match=message):
        generate_basket(lda, mutate(sim), universe, 6)


# ======================= test 15: not enough stocks in a cap category =======================
def test_a_short_cap_category_is_filled_from_another_and_the_shortfall_is_reported():
    result = basket_of(spread_spec(4, 4, 1), 6)  # target 2/2/2, but only one Small Cap stock exists
    assert len(result.basket) == 6
    assert cap_counts(result) == {"Large Cap": 3, "Mid Cap": 2, "Small Cap": 1}
    assert result.target_allocation == {"Large Cap": 2, "Mid Cap": 2, "Small Cap": 2}
    assert result.planned_allocation == {"Large Cap": 3, "Mid Cap": 2, "Small Cap": 1}
    assert (
        "Target was 2 Small Cap stocks, but only 1 eligible Small Cap stock was available. "
        "1 additional stock was selected from Large Cap." in result.notes
    )
    extra_reason = " ".join(result.basket.loc[result.basket["cap_category"] == "Large Cap", "reason"])
    assert "above the target of 2" in extra_reason


def test_a_missing_cap_category_does_not_crash():
    result = basket_of(spread_spec(4, 4, 0), 6)
    assert len(result.basket) == 6 and cap_counts(result) == {"Large Cap": 3, "Mid Cap": 3, "Small Cap": 0}
    assert any("only 0 eligible Small Cap stocks were available" in note and "2 additional stocks were selected from Large Cap (1), Mid Cap (1)" in note for note in result.notes)


def test_a_shortfall_is_never_hidden_even_when_the_basket_is_the_whole_universe():
    result = basket_of(spread_spec(5, 4, 2), 11)  # 4/3/3 style targets, but Small has 2
    assert len(result.basket) == 11 and cap_counts(result) == {"Large Cap": 5, "Mid Cap": 4, "Small Cap": 2}
    assert any(note.startswith("Target was 3 Small Cap stocks, but only 2 eligible Small Cap stocks were available.") for note in result.notes)


def test_no_shortfall_note_when_every_target_is_met():
    assert not [n for n in basket_of(spread_spec(5, 5, 5), 10).notes if n.startswith("Target was")]


# ======================= test 16: output columns =======================
def test_output_columns_are_exactly_as_specified(world):
    result = generate_basket(*world, basket_size=6)
    assert list(result.basket.columns) == ["symbol", "cap_category", "behavior_class", "weight", "allocation", "reason"] == BASKET_COLUMNS
    assert result.basket["weight"].dtype == "float64" and result.basket["allocation"].dtype == "float64"


def test_rows_are_ordered_large_then_mid_then_small_then_by_symbol(world):
    basket = generate_basket(*world, basket_size=10).basket
    order = basket["cap_category"].map(CAP_CATEGORIES.index)
    assert order.is_monotonic_increasing
    for _, group in basket.groupby("cap_category"):
        assert group["symbol"].is_monotonic_increasing
    assert basket.index.tolist() == list(range(10))


# ======================= test 17: determinism =======================
def test_the_same_inputs_always_give_the_same_basket(world):
    first, second = generate_basket(*world, basket_size=7), generate_basket(*world, basket_size=7)
    pd.testing.assert_frame_equal(first.basket, second.basket)
    assert first.notes == second.notes


def test_the_order_of_the_input_rows_does_not_change_the_basket():
    lda, sim, universe = make_world(spread_spec(5, 4, 4), angles=np.linspace(0, 330, 13))
    expected = generate_basket(lda, sim, universe, 8)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(sim))
    shuffled = generate_basket(lda.sample(frac=1, random_state=1), sim.iloc[order, order], universe.sample(frac=1, random_state=2), 8)
    pd.testing.assert_frame_equal(expected.basket, shuffled.basket)
    assert expected.notes == shuffled.notes


def test_ties_are_broken_alphabetically():
    lda, sim, universe = make_world([("B2", "Large Cap", "Defensive"), ("A1", "Large Cap", "Defensive"), ("C3", "Mid Cap", "Defensive"), ("D4", "Small Cap", "Defensive")])
    result = generate_basket(lda, sim, universe, 3)
    assert result.basket.loc[result.basket["cap_category"] == "Large Cap", "symbol"].tolist() == ["A1"]


# ======================= test 18: inputs are not modified =======================
def test_the_input_tables_are_never_modified():
    lda, sim, universe = make_world(CLONE_SPEC + [("SB", "Small Cap", "Balanced")], CLONE_ANGLES + [10])
    before = [t.copy(deep=True) for t in (lda, sim, universe)]
    result = generate_basket(lda, sim, universe, 4)
    basket_statistics(result.basket, sim)
    for original, kept in zip((lda, sim, universe), before):
        pd.testing.assert_frame_equal(original, kept)
    result.basket.loc[0, "symbol"] = "CHANGED"
    pd.testing.assert_frame_equal(universe, before[2])


# ======================= test 19: reasons =======================
def test_every_selected_stock_has_a_reason_built_from_its_actual_selection(world):
    lda, sim, universe = world
    result = generate_basket(lda, sim, universe, basket_size=8)
    behaviour = stock_behaviour(lda)
    for row in result.basket.itertuples(index=False):
        reason = row.reason
        assert isinstance(reason, str) and reason.strip()
        assert reason.startswith(f"{row.cap_category} + {row.behavior_class};")
        assert f"{behaviour.loc[row.symbol, 'share']:.0%} of its days" in reason
        assert "slot" in reason and "pick" in reason
    assert result.basket["reason"].is_unique  # each stock's reason mentions its own pick number


def test_the_reason_quotes_how_consistently_the_stock_had_its_class():
    """A stock that is Defensive on 3 of its 4 days must say 75%, not 100%."""
    spec = [("L1", "Large Cap", "Defensive"), ("M1", "Mid Cap", "Balanced"), ("S1", "Small Cap", "Aggressive")]
    lda, sim, universe = make_world(spec, rows_per_stock=4)
    lda.loc[(lda["symbol"] == "L1") & (lda["date"] == lda["date"].max()), "behavior_class"] = "Balanced"  # L1: 3 Defensive, 1 Balanced
    lda.loc[(lda["symbol"] == "M1").to_numpy() & (lda["date"] <= lda["date"].min() + pd.Timedelta(days=1)).to_numpy(), "behavior_class"] = "Aggressive"  # M1: 2 Aggressive, 2 Balanced: a tie
    reasons = generate_basket(lda, sim, universe, 3).basket.set_index("symbol")["reason"]
    assert "Defensive on 75% of its days" in reasons["L1"]
    assert "Balanced on 50% of its days" in reasons["M1"]  # 2 Aggressive vs 2 Balanced is a tie; ties go to the first class in order (Balanced)
    assert "Aggressive on 100% of its days" in reasons["S1"]


def test_the_first_pick_has_no_similarity_check_and_later_picks_quote_the_real_similarity():
    lda, sim, universe = make_world(CLONE_SPEC, CLONE_ANGLES)
    reasons = generate_basket(lda, sim, universe, 3).basket.set_index("symbol")["reason"]
    assert "pick 1:" in reasons["LA"] and "first pick, so no similarity check was needed" in reasons["LA"]
    assert "the first Aggressive stock in the basket" in reasons["SA"] and "pick 2:" in reasons["SA"]
    assert f"{np.cos(np.radians(240)):.2f} (with LA)" in reasons["SA"]  # -0.50, taken from the matrix
    assert "pick 3:" in reasons["MB"] and "brings Defensive stocks in the basket to 2" in reasons["MB"]
    assert "-0.50 (with" in reasons["MB"] and "within the 0.90 limit" in reasons["MB"]


def test_reasons_mention_the_threshold_that_was_used():
    lda, sim, universe = make_world(CLONE_SPEC, CLONE_ANGLES)
    reasons = generate_basket(lda, sim, universe, 3, similarity_threshold=0.75).basket["reason"]
    assert all("0.75 limit" in r for r in reasons if "first pick" not in r)


# ======================= test 20: basket statistics =======================
def test_basket_statistics_on_a_hand_computed_example():
    basket = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "cap_category": ["Large Cap", "Large Cap", "Small Cap"],
            "behavior_class": ["Defensive", "Defensive", "Aggressive"],
            "weight": 1 / 3, "allocation": 100.0, "reason": "x",
        }
    )
    matrix = pd.DataFrame([[1, 0.9, 0.1], [0.9, 1, -0.3], [0.1, -0.3, 1]], index=list("ABC"), columns=list("ABC"))
    stats = basket_statistics(basket, matrix, threshold=0.5)
    assert stats["average_similarity"] == pytest.approx((0.9 + 0.1 - 0.3) / 3)
    assert stats["maximum_similarity"] == pytest.approx(0.9) and stats["maximum_pair"] == ("A", "B")
    assert stats["n_pairs"] == 3 and stats["pairs_above_threshold"] == 1
    assert stats["behaviour_counts"].to_dict() == {"Defensive": 2, "Balanced": 0, "Aggressive": 1}
    assert stats["behaviour_percent"].to_dict() == pytest.approx({"Defensive": 200 / 3, "Balanced": 0.0, "Aggressive": 100 / 3})
    assert stats["cap_counts"].to_dict() == {"Large Cap": 2, "Mid Cap": 0, "Small Cap": 1}
    assert stats["cap_percent"].to_dict() == pytest.approx({"Large Cap": 200 / 3, "Mid Cap": 0.0, "Small Cap": 100 / 3})
    assert stats["behaviour_counts"].sum() == stats["cap_counts"].sum() == 3
    assert list(stats["behaviour_counts"].index) == CLASS_NAMES and list(stats["cap_counts"].index) == CAP_CATEGORIES


def test_basket_statistics_percentages_sum_to_100_on_a_real_basket(world):
    stats = basket_statistics(generate_basket(*world, basket_size=10).basket, world[1])
    assert stats["behaviour_percent"].sum() == pytest.approx(100) and stats["cap_percent"].sum() == pytest.approx(100)
    assert stats["n_pairs"] == 45


def test_the_pairwise_similarity_table_has_one_row_per_unordered_pair():
    lda, sim, universe = make_world(CLONE_SPEC, CLONE_ANGLES)
    pairs = pairwise_similarities(["SA", "LA", "MB"], sim)
    assert len(pairs) == 3 and set(map(tuple, pairs[["symbol_a", "symbol_b"]].to_numpy())) == {("LA", "MB"), ("LA", "SA"), ("MB", "SA")}
    assert pairs.loc[(pairs["symbol_a"] == "LA") & (pairs["symbol_b"] == "MB"), "similarity"].iloc[0] == pytest.approx(np.cos(np.radians(120)))
    with pytest.raises(ValueError, match="At least two"):
        basket_statistics(generate_basket(*make_world(spread_spec(1, 1, 1)), 3).basket.head(1), make_world(spread_spec(1, 1, 1))[1])


def test_the_similarity_rule_visibly_lowers_the_baskets_similarity():
    lda, sim, universe = make_world(CLONE_SPEC, CLONE_ANGLES)
    with_rule = basket_statistics(generate_basket(lda, sim, universe, 3).basket, sim)
    without_rule = basket_statistics(generate_basket(lda, sim, universe, 3, similarity_threshold=1.0).basket, sim)
    assert with_rule["average_similarity"] < without_rule["average_similarity"]
    assert with_rule["maximum_similarity"] < without_rule["maximum_similarity"]


# ======================= validate_basket: the final safety checks can really fail =======================
@pytest.fixture
def good_basket(world):
    return generate_basket(*world, basket_size=6).basket


@pytest.mark.parametrize(
    "tamper, message",
    [
        (lambda b: b.drop(columns=["reason"]), "columns must be"),
        (lambda b: b.iloc[:-1], "expected 6"),
        (lambda b: b.assign(symbol=["L01"] * 6), "duplicate stocks"),
        (lambda b: b.assign(symbol=b["symbol"].str.replace("L", "Q", regex=False)), "not in the universe"),
        (lambda b: b.assign(cap_category="Small Cap"), "does not match the universe"),
        (lambda b: b.assign(behavior_class="Neutral"), "invalid behaviour class"),
        (lambda b: b.assign(weight=0.2), "weights sum"),
        (lambda b: b.assign(allocation=b["allocation"] + 1), "allocations sum"),
        (lambda b: b.assign(reason=""), "needs a reason"),
    ],
)
def test_validate_basket_catches_broken_baskets(world, good_basket, tamper, message):
    validate_basket(good_basket, world[2], 6, 100_000)  # the untouched basket passes
    with pytest.raises(ValueError, match=message):
        validate_basket(tamper(good_basket), world[2], 6, 100_000)


# ======================= saving, plotting, rupees =======================
@pytest.mark.parametrize("amount, text", [(100_000, "₹100,000"), (8333.3333, "₹8,333.33"), (12.5, "₹12.50"), (1e6, "₹1,000,000"), (50_000.0, "₹50,000")])
def test_rupee_formatting(amount, text):
    assert format_rupees(amount) == text


def test_saved_csv_has_the_required_columns_and_round_trips(world, tmp_path):
    basket = generate_basket(*world, basket_size=7, capital=50_000).basket
    path = save_basket(basket, tmp_path / "nested" / "basket.csv")
    saved = pd.read_csv(path)
    assert list(saved.columns) == BASKET_COLUMNS and len(saved) == 7
    assert saved["symbol"].tolist() == basket["symbol"].tolist()
    assert saved["reason"].tolist() == basket["reason"].tolist()  # semicolons and commas survive
    np.testing.assert_allclose(saved["allocation"], basket["allocation"])
    assert saved["allocation"].sum() == pytest.approx(50_000)


def test_allocation_chart_is_written(world, tmp_path):
    basket = generate_basket(*world, basket_size=6, capital=50_000).basket
    path = plot_basket_allocation(basket, tmp_path / "plots" / "basket.png", 50_000, note="test note")
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    with pytest.raises(ValueError, match="empty"):
        plot_basket_allocation(basket.head(0), tmp_path / "none.png", 50_000)


# ======================= the printed report =======================
def test_report_shows_the_required_sections_and_the_computed_numbers(world, capsys):
    result = generate_basket(*world, basket_size=10, capital=100_000)
    stats = print_basket_report(result, world[1], source="data/raw/dev_prices_synthetic.csv", eligible_symbols=sorted(world[2]["symbol"]))
    text = capsys.readouterr().out
    for expected in [
        "=== INTELLIGENT STOCK BASKET ===",
        DEVELOPMENT_NOTICE,
        "The current dataset is synthetic and is used only to verify that the implementation works.",
        "Capital: ₹100,000",
        "Basket size: 10",
        "Target cap allocation:",
        "Large Cap: 4", "Mid Cap:   3", "Small Cap: 3",
        "Actual basket:",
        "Symbol     Cap        Behaviour   Weight   Allocation",
        "Total weight: 100.00%",
        "Total allocation: ₹100,000",
        "Behaviour distribution:",
        "Cap distribution:",
        "Average pairwise similarity:",
        "Maximum pairwise similarity:",
        "Selection notes:",
        "Why each stock was selected:",
        METHOD_STATEMENT,
        DISCLAIMER,
    ]:
        assert expected in text
    assert DISCLAIMER == "This basket is generated from synthetic development data and is not an investment recommendation."
    assert f"{stats['average_similarity']:.4f}" in text and f"{stats['maximum_similarity']:.4f}" in text
    for row in result.basket.itertuples(index=False):
        assert row.symbol in text and row.reason in text
    assert "better" not in METHOD_STATEMENT.lower().replace("not yet been tested against a benchmark", "")  # no performance claim


def test_report_prints_the_shortfall_notes(capsys):
    lda, sim, universe = make_world(spread_spec(4, 4, 1))
    result = generate_basket(lda, sim, universe, 6)
    print_basket_report(result, sim)
    text = capsys.readouterr().out
    assert "- Target was 2 Small Cap stocks, but only 1 eligible Small Cap stock was available." in text
    assert "synthetic" not in text.split("Capital:")[0].lower().replace("synthetic development cap classifications", "")


# ======================= command line =======================
@pytest.fixture
def demo_files(tmp_path):
    symbols = get_development_universe()["symbol"].tolist()
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((10, 4))
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    pd.DataFrame(unit @ unit.T, index=symbols, columns=symbols).to_csv(tmp_path / "sim.csv")
    classes = [CLASS_NAMES[i % 3] for i in range(10)]
    pd.DataFrame(
        [{"date": "2024-01-0%d" % (k + 1), "symbol": s, "behavior_class": c, "LD1": 0.0, "LD2": 0.0} for s, c in zip(symbols, classes) for k in range(4)]
    ).to_csv(tmp_path / "lda.csv", index=False)
    config = {
        "data": {"prices_file": "data/raw/dev_prices_synthetic.csv"},
        "lda": {"output_file": str(tmp_path / "lda.csv")},
        "similarity": {"matrix_file": str(tmp_path / "sim.csv")},
        "basket": {"basket_size": 6, "capital": 50_000, "similarity_threshold": 0.9,
                   "output_file": str(tmp_path / "out" / "basket.csv"), "plot_file": str(tmp_path / "plots" / "basket.png")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def run_main(monkeypatch, config_path, *args):
    monkeypatch.setattr(sys, "argv", ["basket", "--config", str(config_path), *args])
    main()


def test_command_line_uses_the_config_defaults_and_writes_the_files(tmp_path, demo_files, monkeypatch, capsys):
    run_main(monkeypatch, demo_files)
    text = capsys.readouterr().out
    assert "Capital: ₹50,000" in text and "Basket size: 6" in text and "Total allocation: ₹50,000" in text
    assert DISCLAIMER in text
    saved = pd.read_csv(tmp_path / "out" / "basket.csv")
    assert list(saved.columns) == BASKET_COLUMNS and len(saved) == 6
    assert saved["allocation"].sum() == pytest.approx(50_000) and saved["weight"].sum() == pytest.approx(1)
    assert (tmp_path / "plots" / "basket.png").read_bytes()[:8] == PNG_SIGNATURE


def test_command_line_flags_override_the_config(tmp_path, demo_files, monkeypatch, capsys):
    run_main(monkeypatch, demo_files, "--basket-size", "10", "--capital", "100000", "--similarity-threshold", "0.8")
    text = capsys.readouterr().out
    assert "Capital: ₹100,000" in text and "Basket size: 10" in text and "Similarity limit: 0.80" in text
    saved = pd.read_csv(tmp_path / "out" / "basket.csv")
    assert len(saved) == 10 and saved["allocation"].sum() == pytest.approx(100_000)
    assert saved["symbol"].is_unique


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--basket-size", "2"], "at least 3"),
        (["--basket-size", "11"], "larger than the number of eligible stocks"),
        (["--capital", "-5"], "capital must be a positive number"),
        (["--capital", "0"], "capital must be a positive number"),
        (["--similarity-threshold", "1.5"], "similarity_threshold"),
    ],
)
def test_command_line_fails_clearly_on_invalid_input(demo_files, monkeypatch, capsys, flags, message):
    with pytest.raises(SystemExit) as exit_info:
        run_main(monkeypatch, demo_files, *flags)
    assert exit_info.value.code == 2 and message in capsys.readouterr().err


def test_command_line_says_which_earlier_step_to_run_when_an_input_is_missing(tmp_path, demo_files, monkeypatch, capsys):
    (tmp_path / "sim.csv").unlink()
    with pytest.raises(SystemExit):
        run_main(monkeypatch, demo_files)
    assert "python -m src.similarity" in capsys.readouterr().err
    (tmp_path / "lda.csv").unlink()
    with pytest.raises(SystemExit):
        run_main(monkeypatch, demo_files)
    assert "python -m src.lda_model" in capsys.readouterr().err


def test_config_has_the_basket_settings():
    cfg = load_config(PROJECT_ROOT / "config.yaml")["basket"]
    assert cfg["basket_size"] == 10 and cfg["capital"] == 100000 and cfg["similarity_threshold"] == 0.90
    assert cfg["output_file"].endswith("basket.csv") and cfg["plot_file"].endswith("basket_allocation.png")


# ======================= the development data, end to end =======================
@pytest.fixture(scope="module")
def dev_inputs():
    config = load_config(PROJECT_ROOT / "config.yaml")
    stocks = get_provider(config, PROJECT_ROOT).get_historical_data()
    market = LocalDataProvider(PROJECT_ROOT / config["data"]["market_index_file"]).get_historical_data()
    features = clean_feature_data(build_features(stocks, market))
    pca = transform_pca(features, fit_pca(features, n_components=5))
    similarity = calculate_similarity(create_stock_profiles(pca))
    labels = create_behavior_labels(features, fit_label_rules(features))
    lda = transform_lda(features, fit_lda(features, labels), labels)
    return lda, similarity, get_development_universe()


@pytest.mark.parametrize("n, capital", [(6, 50_000), (10, 100_000), (3, 9_999), (8, 1_000_000)])
def test_basket_from_the_whole_phase_2_to_5_pipeline(dev_inputs, n, capital):
    lda, similarity, universe = dev_inputs
    before = [t.copy(deep=True) for t in dev_inputs]
    result = generate_basket(lda, similarity, universe, n, capital)
    basket = result.basket
    assert len(basket) == n and basket["symbol"].is_unique
    assert basket["weight"].sum() == pytest.approx(1) and basket["allocation"].sum() == pytest.approx(capital)
    assert tuple(cap_counts(result).values()) == tuple(cap_targets(n).values())  # the dev universe has 4/3/3, so every target can be met
    assert set(basket["behavior_class"]) <= set(CLASS_NAMES)
    assert all(r for r in basket["reason"])
    validate_basket(basket, universe, n, capital)
    for original, kept in zip(dev_inputs, before):
        pd.testing.assert_frame_equal(original, kept)


def test_selecting_the_whole_dev_universe_reports_the_similarity_it_could_not_avoid(dev_inputs):
    lda, similarity, universe = dev_inputs
    result = generate_basket(lda, similarity, universe, 10)
    assert set(result.basket["symbol"]) == set(universe["symbol"])
    stats = basket_statistics(result.basket, similarity)
    flagged = [note for note in result.notes if "was selected even though" in note]
    if stats["maximum_similarity"] > 0.90:  # some pair is above the limit, so the basket must admit it
        assert flagged
    else:
        assert not flagged
    for note in flagged:
        assert "above the 0.90 limit" in note
