"""
Tests for Phase 5 (src/similarity.py).

Most tests use tiny hand-made tables where the right answer can be worked out on paper
(means of a few numbers, cosine of simple vectors). A few run the real Phase 2 -> Phase 3
pipeline on the synthetic development data. They check that the code is correct; they say
nothing about real markets, because the data is random.

Run:  pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import build_features, clean_feature_data
from src.pca_model import fit_pca, transform_pca
from src.similarity import (
    DISCLAIMER,
    calculate_similarity,
    create_stock_profiles,
    find_similar_stocks,
    main,
    pc_columns,
    plot_similar_stocks,
    print_similarity_report,
    save_similarity_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ======================= helpers =======================
def profiles_from(vectors: dict) -> pd.DataFrame:
    """A stock-profile table (symbol, PC1, PC2, ...) from {symbol: [pc values]}."""
    k = len(next(iter(vectors.values())))
    return pd.DataFrame(
        [{"symbol": s, **{f"PC{i + 1}": v for i, v in enumerate(values)}} for s, values in vectors.items()],
        columns=["symbol"] + [f"PC{i + 1}" for i in range(k)],
    )


def small_pca_data() -> pd.DataFrame:
    """Several rows per stock, with means that are easy to compute by hand."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02", "2024-01-01"]),
            "symbol": ["AAA", "AAA", "AAA", "BBB", "BBB", "CCC"],
            "PC1": [1.0, 3.0, 5.0, 10.0, 20.0, -4.0],
            "PC2": [0.0, 2.0, 4.0, -2.0, -4.0, 8.0],
            "PC3": [1.0, 1.0, 1.0, 0.5, 1.5, 0.0],
        }
    )


def random_profiles(n=7, k=5, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return profiles_from({f"S{i:02d}": rng.standard_normal(k).tolist() for i in range(n)})


def sym(df):
    return df["symbol"].tolist()


# ======================= test 1: profiles are means =======================
def test_profile_values_are_the_means_of_each_component():
    profiles = create_stock_profiles(small_pca_data()).set_index("symbol")
    assert profiles.loc["AAA"].tolist() == pytest.approx([3.0, 2.0, 1.0])  # (1+3+5)/3, (0+2+4)/3, (1+1+1)/3
    assert profiles.loc["BBB"].tolist() == pytest.approx([15.0, -3.0, 1.0])
    assert profiles.loc["CCC"].tolist() == pytest.approx([-4.0, 8.0, 0.0])  # a single row is its own mean


def test_profiles_have_symbol_and_pc_columns_only():
    profiles = create_stock_profiles(small_pca_data())
    assert list(profiles.columns) == ["symbol", "PC1", "PC2", "PC3"]  # no date
    assert profiles["PC1"].dtype == "float64"


# ======================= test 2: one profile per stock =======================
def test_one_row_per_symbol():
    data = small_pca_data()
    profiles = create_stock_profiles(data)
    assert len(profiles) == data["symbol"].nunique() == 3
    assert profiles["symbol"].is_unique
    assert sym(profiles) == ["AAA", "BBB", "CCC"]


def test_row_order_of_the_input_does_not_change_the_profiles():
    data = small_pca_data()
    shuffled = data.sample(frac=1, random_state=4)
    pd.testing.assert_frame_equal(create_stock_profiles(data), create_stock_profiles(shuffled))


def test_pc_columns_are_found_in_numerical_order():
    table = pd.DataFrame(columns=["symbol", "PC10", "PC2", "PC1", "close", "PCX", "xPC3"])
    assert pc_columns(table) == ["PC1", "PC2", "PC10"]
    wide = pd.DataFrame({"symbol": ["A", "B"], **{f"PC{i}": [1.0, 2.0] for i in range(1, 13)}})
    assert pc_columns(create_stock_profiles(wide)) == [f"PC{i}" for i in range(1, 13)]


def test_a_profile_does_not_depend_on_other_stocks_rows():
    data = small_pca_data()
    changed = data.copy()
    changed.loc[changed["symbol"] == "BBB", ["PC1", "PC2", "PC3"]] += 100
    before, after = create_stock_profiles(data).set_index("symbol"), create_stock_profiles(changed).set_index("symbol")
    pd.testing.assert_frame_equal(before.loc[["AAA", "CCC"]], after.loc[["AAA", "CCC"]])
    assert not before.loc["BBB"].equals(after.loc["BBB"])
    assert calculate_similarity(create_stock_profiles(data)).loc["AAA", "CCC"] == pytest.approx(
        calculate_similarity(create_stock_profiles(changed)).loc["AAA", "CCC"]
    )


# ======================= tests 3, 4, 5: the similarity matrix =======================
@pytest.mark.parametrize("n", [1, 2, 7, 25])
def test_matrix_is_n_by_n_with_matching_rows_and_columns(n):
    profiles = random_profiles(n=n)
    matrix = calculate_similarity(profiles)
    assert matrix.shape == (n, n)
    assert list(matrix.index) == list(matrix.columns) == sym(profiles)


def test_diagonal_is_one():
    matrix = calculate_similarity(random_profiles(n=12))
    np.testing.assert_allclose(np.diag(matrix.to_numpy()), 1.0, atol=1e-12)


def test_matrix_is_symmetric_and_within_minus_one_and_one():
    matrix = calculate_similarity(random_profiles(n=15, seed=3)).to_numpy()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert matrix.min() >= -1.0 and matrix.max() <= 1.0


def test_similarity_of_a_pair_is_the_same_in_both_directions():
    matrix = calculate_similarity(random_profiles())
    assert matrix.loc["S01", "S04"] == pytest.approx(matrix.loc["S04", "S01"], abs=1e-12)


# ======================= test 6: known vectors =======================
def test_identical_and_opposite_vectors():
    matrix = calculate_similarity(profiles_from({"A": [1, 0, 0], "B": [1, 0, 0], "C": [-1, 0, 0]}))
    assert matrix.loc["A", "B"] == pytest.approx(1.0)
    assert matrix.loc["A", "C"] == pytest.approx(-1.0)
    assert matrix.loc["B", "C"] == pytest.approx(-1.0)


def test_perpendicular_and_45_degree_vectors_and_a_hand_calculation():
    matrix = calculate_similarity(profiles_from({"A": [1, 0], "B": [0, 1], "C": [1, 1], "D": [3, 4], "E": [4, 3]}))
    assert matrix.loc["A", "B"] == pytest.approx(0.0, abs=1e-12)  # 90 degrees
    assert matrix.loc["A", "C"] == pytest.approx(1 / np.sqrt(2))  # 45 degrees: 1 / (1 x sqrt 2)
    assert matrix.loc["D", "E"] == pytest.approx(24 / 25)  # (3x4 + 4x3) / (5 x 5)


def test_only_direction_matters_not_length():
    matrix = calculate_similarity(profiles_from({"A": [1.0, 2.0, -1.0], "B": [3.0, 6.0, -3.0], "C": [0.01, 0.02, -0.01]}))
    assert matrix.loc["A", "B"] == pytest.approx(1.0)
    assert matrix.loc["A", "C"] == pytest.approx(1.0)


def test_matrix_matches_the_textbook_formula_on_random_profiles():
    profiles = random_profiles(n=6, k=5, seed=9)
    matrix = calculate_similarity(profiles)
    vectors = profiles[pc_columns(profiles)].to_numpy()
    for i in range(6):
        for j in range(6):
            a, b = vectors[i], vectors[j]
            assert matrix.iloc[i, j] == pytest.approx(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ======================= tests 7, 8, 9: finding similar stocks =======================
@pytest.fixture
def ladder():
    """Stocks at known angles from A: B closest, then C, D, E, F (opposite)."""
    angles = {"A": 0, "B": 10, "C": 30, "D": 60, "E": 100, "F": 180}
    return profiles_from({s: [np.cos(np.radians(a)), np.sin(np.radians(a))] for s, a in angles.items()})


def test_the_stock_itself_is_never_in_its_own_results(ladder):
    for symbol in sym(ladder):
        assert symbol not in sym(find_similar_stocks(ladder, symbol, top_n=10))
    assert symbol not in sym(find_similar_stocks(random_profiles(), "S03", top_n=100))


def test_results_are_sorted_from_most_to_least_similar(ladder):
    result = find_similar_stocks(ladder, "A", top_n=5)
    assert sym(result) == ["B", "C", "D", "E", "F"]
    assert result["similarity"].is_monotonic_decreasing
    assert result["similarity"].tolist() == pytest.approx([np.cos(np.radians(a)) for a in (10, 30, 60, 100, 180)])
    assert result["rank"].tolist() == [1, 2, 3, 4, 5]
    assert list(result.columns) == ["rank", "symbol", "similarity"]


def test_ranking_follows_the_matrix_on_random_data():
    profiles = random_profiles(n=9, seed=2)
    result = find_similar_stocks(profiles, "S05", top_n=8)
    expected = calculate_similarity(profiles).loc["S05"].drop("S05").sort_values(ascending=False)
    assert sym(result) == expected.index.tolist()
    assert result["similarity"].to_numpy() == pytest.approx(expected.to_numpy())


def test_top_n_limits_the_number_of_results(ladder):
    assert len(find_similar_stocks(ladder, "A", top_n=3)) == 3
    assert sym(find_similar_stocks(ladder, "A", top_n=3)) == ["B", "C", "D"]
    assert len(find_similar_stocks(ladder, "A", top_n=1)) == 1
    assert len(find_similar_stocks(ladder, "A")) == 5  # the default is 5


def test_asking_for_more_than_exists_returns_all_the_alternatives(ladder):
    assert len(find_similar_stocks(ladder, "A", top_n=50)) == 5  # 6 stocks, minus itself
    only_two = profiles_from({"A": [1, 0], "B": [1, 1]})
    assert sym(find_similar_stocks(only_two, "A", top_n=5)) == ["B"]
    alone = find_similar_stocks(profiles_from({"A": [1, 0]}), "A", top_n=5)
    assert len(alone) == 0 and list(alone.columns) == ["rank", "symbol", "similarity"]


def test_ties_are_broken_alphabetically_so_results_are_reproducible():
    tied = profiles_from({"A": [1, 0], "Z": [0, 1], "M": [0, 2], "B": [0, 3]})  # all three have similarity 0 to A
    assert sym(find_similar_stocks(tied, "A", top_n=3)) == ["B", "M", "Z"]


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "3", None])
def test_invalid_top_n_is_rejected(ladder, bad):
    with pytest.raises(ValueError, match="top_n"):
        find_similar_stocks(ladder, "A", top_n=bad)


# ======================= test 10: unknown symbol =======================
def test_unknown_symbol_raises_a_clear_error(ladder):
    with pytest.raises(ValueError, match="Unknown symbol 'ZZZ'.*Available symbols: A, B, C, D, E, F"):
        find_similar_stocks(ladder, "ZZZ")
    with pytest.raises(ValueError, match="Unknown symbol"):
        find_similar_stocks(ladder, "a")  # symbols are matched exactly


# ======================= test 11: the original data is not modified =======================
def test_the_pca_table_and_profiles_are_never_modified():
    data = small_pca_data()
    data_before = data.copy(deep=True)
    profiles = create_stock_profiles(data)
    profiles_before = profiles.copy(deep=True)
    matrix = calculate_similarity(profiles)
    find_similar_stocks(profiles, "AAA", top_n=2)
    pd.testing.assert_frame_equal(data, data_before)
    pd.testing.assert_frame_equal(profiles, profiles_before)
    matrix.iloc[0, 0] = 99.0  # the returned matrix is a new object, not a view of the profiles
    pd.testing.assert_frame_equal(profiles, profiles_before)


# ======================= test 12: missing or invalid values =======================
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_missing_or_infinite_pca_values_are_rejected(bad):
    data = small_pca_data()
    data.loc[1, "PC2"] = bad
    with pytest.raises(ValueError, match="missing, non-numeric or infinite.*AAA"):
        create_stock_profiles(data)


def test_non_numeric_pca_values_are_rejected():
    data = small_pca_data().astype({"PC1": object})
    data.loc[0, "PC1"] = "n/a"
    with pytest.raises(ValueError, match="non-numeric"):
        create_stock_profiles(data)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_profiles_with_invalid_values_are_rejected_by_the_similarity_functions(bad):
    profiles = random_profiles()
    profiles.loc[2, "PC3"] = bad
    with pytest.raises(ValueError, match="similarity would be wrong"):
        calculate_similarity(profiles)
    with pytest.raises(ValueError, match="similarity would be wrong"):
        find_similar_stocks(profiles, "S00")


def test_an_all_zero_profile_is_rejected_instead_of_getting_a_made_up_similarity():
    profiles = profiles_from({"A": [1, 0, 0], "FLAT": [0, 0, 0], "C": [0, 1, 0]})
    with pytest.raises(ValueError, match="undefined for an all-zero profile.*FLAT"):
        calculate_similarity(profiles)


def test_malformed_tables_are_rejected():
    with pytest.raises(ValueError, match="'symbol' column"):
        create_stock_profiles(small_pca_data().drop(columns=["symbol"]))
    with pytest.raises(ValueError, match="no principal-component columns"):
        create_stock_profiles(small_pca_data()[["date", "symbol"]])
    with pytest.raises(ValueError, match="empty"):
        create_stock_profiles(small_pca_data().head(0))
    with pytest.raises(ValueError, match="without a symbol"):
        create_stock_profiles(small_pca_data().assign(symbol=[None, "AAA", "AAA", "BBB", "BBB", "CCC"]))
    with pytest.raises(ValueError, match="one row per symbol.*A"):
        calculate_similarity(profiles_from({"A": [1, 0], "B": [0, 1]}).pipe(lambda d: pd.concat([d, d.head(1)])))
    with pytest.raises(ValueError, match="symbol"):
        calculate_similarity(random_profiles().drop(columns=["symbol"]))
    with pytest.raises(ValueError, match="no principal-component"):
        calculate_similarity(random_profiles()[["symbol"]])
    with pytest.raises(ValueError, match="empty"):
        calculate_similarity(random_profiles().head(0))


# ======================= leakage: profiles are built from whatever rows you give them =======================
def test_profiles_from_a_date_truncated_table_do_not_see_later_rows():
    """
    How a backtest must use this: cut the table at the rebalance date FIRST, then build profiles.
    Later rows then cannot influence them. (Today's demo uses the full period, which is exploratory only.)
    """
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2023-01-02", periods=60)
    data = pd.DataFrame(
        {
            "date": np.tile(dates, 3),
            "symbol": np.repeat(["AAA", "BBB", "CCC"], 60),
            **{f"PC{i}": rng.standard_normal(180) for i in (1, 2, 3)},
        }
    )
    cutoff = dates[39]
    changed = data.copy()
    changed.loc[changed["date"] > cutoff, ["PC1", "PC2", "PC3"]] += 50  # wreck the future

    point_in_time = lambda d: create_stock_profiles(d[d["date"] <= cutoff])
    pd.testing.assert_frame_equal(point_in_time(data), point_in_time(changed))
    assert not create_stock_profiles(data).equals(create_stock_profiles(changed))  # the full-period profile DOES see the future


# ======================= saving, plotting, report =======================
def test_saved_matrix_round_trips(tmp_path):
    matrix = calculate_similarity(random_profiles(n=6))
    path = save_similarity_matrix(matrix, tmp_path / "nested" / "similarity_matrix.csv")
    saved = pd.read_csv(path, index_col=0)
    assert saved.shape == (6, 6)
    assert list(saved.index) == list(saved.columns) == list(matrix.index)
    np.testing.assert_allclose(saved.to_numpy(), matrix.to_numpy(), atol=1e-12)


def test_bar_chart_is_written_for_positive_and_negative_similarities(tmp_path, ladder):
    mixed = find_similar_stocks(ladder, "A", top_n=5)  # includes a negative value (F is opposite)
    assert mixed["similarity"].min() < 0
    path = plot_similar_stocks(mixed, "A", tmp_path / "plots" / "sim_A.png", note="test note")
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    only_positive = find_similar_stocks(ladder, "A", top_n=3)
    assert plot_similar_stocks(only_positive, "A", tmp_path / "pos.png").exists()
    with pytest.raises(ValueError, match="no similar stocks"):
        plot_similar_stocks(find_similar_stocks(profiles_from({"A": [1, 0]}), "A"), "A", tmp_path / "none.png")


def test_report_shows_the_fitted_numbers_and_the_required_statements(capsys):
    profiles = random_profiles(n=8, k=5)
    similar = find_similar_stocks(profiles, "S02", top_n=5)
    print_similarity_report(profiles, "S02", similar, "out/similarity_matrix.csv", source="data/raw/dev_prices_synthetic.csv")
    text = capsys.readouterr().out
    for expected in [
        "=== STOCK BEHAVIOURAL SIMILARITY ===",
        "PCA components used: 5",
        "Number of stocks: 8",
        "Selected stock: S02",
        "Most similar stocks:",
        "Rank  Symbol",
        "out/similarity_matrix.csv",
        "The current dataset is synthetic and is used only to verify that the implementation works.",
        "Similarity represents historical behavioural similarity in PCA space. It does not represent expected future returns.",
        "full period",
    ]:
        assert expected in text
    assert DISCLAIMER in text
    for row in similar.itertuples(index=False):
        assert f"{row.symbol}" in text and f"{row.similarity:.4f}" in text
    assert "S02" not in text.split("Most similar stocks:")[1].split("Similarity matrix:")[0]


def test_the_synthetic_notice_only_appears_for_synthetic_data(capsys):
    profiles = random_profiles()
    print_similarity_report(profiles, "S00", find_similar_stocks(profiles, "S00"), "m.csv", source="data/raw/real_prices.csv")
    text = capsys.readouterr().out
    assert "synthetic" not in text.lower()
    assert DISCLAIMER in text  # the disclaimer is always printed


# ======================= config =======================
def test_config_has_the_similarity_settings():
    cfg = load_config(PROJECT_ROOT / "config.yaml")["similarity"]
    assert cfg["matrix_file"].endswith("similarity_matrix.csv") and cfg["plots_dir"]


# ======================= the development data, end to end =======================
@pytest.fixture(scope="module")
def dev_pca_data():
    config = load_config(PROJECT_ROOT / "config.yaml")
    stocks = get_provider(config, PROJECT_ROOT).get_historical_data()
    market = LocalDataProvider(PROJECT_ROOT / config["data"]["market_index_file"]).get_historical_data()
    features = clean_feature_data(build_features(stocks, market))
    return transform_pca(features, fit_pca(features, n_components=5))


def test_similarity_runs_on_the_phase3_output(dev_pca_data):
    original = dev_pca_data.copy(deep=True)
    profiles = create_stock_profiles(dev_pca_data)
    n = dev_pca_data["symbol"].nunique()
    assert len(profiles) == n and list(profiles.columns) == ["symbol", "PC1", "PC2", "PC3", "PC4", "PC5"]

    matrix = calculate_similarity(profiles)
    assert matrix.shape == (n, n)
    np.testing.assert_allclose(np.diag(matrix.to_numpy()), 1.0, atol=1e-12)
    np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T, atol=1e-12)

    first = profiles["symbol"].iloc[0]
    similar = find_similar_stocks(profiles, first, top_n=5)
    assert len(similar) == min(5, n - 1) and first not in sym(similar)
    assert similar["similarity"].is_monotonic_decreasing
    pd.testing.assert_frame_equal(dev_pca_data, original)  # Phase 3 output untouched


def test_profile_means_agree_with_a_plain_groupby_on_the_dev_data(dev_pca_data):
    profiles = create_stock_profiles(dev_pca_data).set_index("symbol")
    expected = dev_pca_data.groupby("symbol")[["PC1", "PC2", "PC3", "PC4", "PC5"]].mean()
    pd.testing.assert_frame_equal(profiles, expected, check_names=False)


# ======================= command-line demo =======================
@pytest.fixture
def demo_config(tmp_path):
    pca = small_pca_data().assign(PC4=0.5, PC5=[0.1, 0.2, 0.3, -0.1, -0.2, 0.4])
    pca.to_csv(tmp_path / "pca.csv", index=False)
    config = {
        "data": {"prices_file": "data/raw/dev_prices_synthetic.csv"},
        "pca": {"output_file": str(tmp_path / "pca.csv")},
        "similarity": {"matrix_file": str(tmp_path / "out" / "matrix.csv"), "plots_dir": str(tmp_path / "plots")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_command_line_demo_selects_the_first_symbol_and_writes_files(tmp_path, demo_config, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["similarity", "--config", str(demo_config)])
    main()
    text = capsys.readouterr().out
    assert "Selected stock: AAA" in text  # no symbol given: the first one available
    assert "PCA components used: 5" in text and "Number of stocks: 3" in text
    assert DISCLAIMER in text
    saved = pd.read_csv(tmp_path / "out" / "matrix.csv", index_col=0)
    assert saved.shape == (3, 3) and list(saved.index) == ["AAA", "BBB", "CCC"]
    assert (tmp_path / "plots" / "similarity_AAA.png").read_bytes()[:8] == PNG_SIGNATURE
    ranked = text.split("Most similar stocks:")[1].split("Similarity matrix:")[0]
    assert "AAA" not in ranked.replace("Rank  Symbol", "")  # not in its own list


def test_command_line_demo_accepts_a_symbol_and_top_n(tmp_path, demo_config, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["similarity", "--config", str(demo_config), "--symbol", "CCC", "--top-n", "1"])
    main()
    text = capsys.readouterr().out
    assert "Selected stock: CCC" in text
    ranked_lines = [line for line in text.split("Most similar stocks:")[1].split("Similarity matrix:")[0].splitlines()[2:] if line.strip()]
    assert len(ranked_lines) == 1
    assert (tmp_path / "plots" / "similarity_CCC.png").exists()


def test_command_line_demo_fails_clearly_on_bad_input(tmp_path, demo_config, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["similarity", "--config", str(demo_config), "--symbol", "NOPE"])
    with pytest.raises(SystemExit) as unknown:
        main()
    assert unknown.value.code == 2 and "Unknown symbol 'NOPE'" in capsys.readouterr().err

    (tmp_path / "pca.csv").unlink()
    monkeypatch.setattr(sys, "argv", ["similarity", "--config", str(demo_config)])
    with pytest.raises(SystemExit) as missing:
        main()
    assert missing.value.code == 2 and "python -m src.pca_model" in capsys.readouterr().err
