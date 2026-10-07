"""GET /api/analysis/pca and GET /api/analysis/lda (real PCA and LDA on the synthetic data)."""

import numpy as np
import pytest

from src.labels import CLASS_NAMES
from src.lda_model import get_lda_explained_variance, get_lda_loadings
from src.pca_model import get_explained_variance, get_feature_loadings

from conftest import strict_json


@pytest.fixture(scope="module")
def pca(client):
    return client.get("/api/analysis/pca").json()


@pytest.fixture(scope="module")
def lda(client):
    return client.get("/api/analysis/lda").json()


# ===================== PCA =====================
def test_pca_structure(pca):
    assert set(pca) == {"components", "component_names", "explained_variance", "cumulative_variance", "loadings", "observations",
                        "total_observations", "returned_observations", "n_training_rows", "mode", "notice"}
    assert pca["components"] == 5 and pca["component_names"] == ["PC1", "PC2", "PC3", "PC4", "PC5"]
    assert pca["mode"] == "exploratory_full_history" and "not a performance test" in pca["notice"]


def test_pca_explained_variance_is_real_and_consistent(pca, engine):
    expected = get_explained_variance(engine.pca)
    assert pca["explained_variance"] == pytest.approx(expected["explained_variance_ratio"].tolist())
    assert pca["cumulative_variance"] == pytest.approx(expected["cumulative_explained_variance"].tolist())
    assert len(pca["explained_variance"]) == len(pca["cumulative_variance"]) == 5
    assert pca["explained_variance"] == sorted(pca["explained_variance"], reverse=True)  # PC1 explains the most
    assert pca["cumulative_variance"][-1] == pytest.approx(sum(pca["explained_variance"]))
    assert 0 < pca["cumulative_variance"][-1] <= 1
    assert all(b > a for a, b in zip(pca["cumulative_variance"], pca["cumulative_variance"][1:]))


def test_pca_loadings_are_the_real_feature_weights(pca, engine):
    expected = get_feature_loadings(engine.pca)
    assert [row["feature"] for row in pca["loadings"]] == list(expected.index) and len(pca["loadings"]) == 13
    for row in pca["loadings"]:
        assert list(row["loadings"]) == ["PC1", "PC2", "PC3", "PC4", "PC5"]
        assert [row["loadings"][c] for c in expected.columns] == pytest.approx(expected.loc[row["feature"]].tolist())
    for component in ("PC1", "PC2"):  # each component has unit length
        assert sum(row["loadings"][component] ** 2 for row in pca["loadings"]) == pytest.approx(1.0)


def test_pca_observations_are_real_scores(pca, engine):
    assert pca["total_observations"] == len(engine.scores) == 20280
    assert pca["returned_observations"] == len(pca["observations"]) == 2000
    scores = engine.scores.set_index(["date", "symbol"])
    for observation in pca["observations"][:50]:
        assert set(observation) == {"date", "symbol", "scores"} and len(observation["date"]) == 10
        row = scores.loc[(np.datetime64(observation["date"]), observation["symbol"])]
        assert [observation["scores"][c] for c in ("PC1", "PC2", "PC3", "PC4", "PC5")] == pytest.approx(row[["PC1", "PC2", "PC3", "PC4", "PC5"]].tolist())


def test_pca_observations_are_sorted_and_reproducible(client, pca):
    keys = [(o["date"], o["symbol"]) for o in pca["observations"]]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert client.get("/api/analysis/pca").json() == pca


def test_pca_max_points_controls_the_sample_size(client):
    small = client.get("/api/analysis/pca?max_points=10").json()
    assert small["returned_observations"] == len(small["observations"]) == 10 and small["total_observations"] == 20280
    everything = client.get("/api/analysis/pca?max_points=50000").json()
    assert everything["returned_observations"] == 20280


def test_pca_components_parameter(client, engine):
    three = client.get("/api/analysis/pca?components=3").json()
    assert three["components"] == 3 and len(three["explained_variance"]) == 3 and three["component_names"] == ["PC1", "PC2", "PC3"]
    assert len(three["loadings"][0]["loadings"]) == 3
    # the first components do not depend on how many are kept
    five = client.get("/api/analysis/pca").json()
    assert three["explained_variance"] == pytest.approx(five["explained_variance"][:3])


@pytest.mark.parametrize("query", ["components=0", "components=14", "components=abc", "max_points=0", "max_points=50001", "max_points=-3"])
def test_pca_rejects_bad_parameters(client, query):
    response = client.get(f"/api/analysis/pca?{query}")
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"


def test_pca_response_is_strict_json_with_python_types(client):
    body = strict_json(client.get("/api/analysis/pca?max_points=100"))
    assert type(body["components"]) is int and type(body["explained_variance"][0]) is float


# ===================== LDA =====================
def test_lda_structure(lda):
    assert set(lda) == {"components", "classes", "training_accuracy", "majority_baseline", "explained_variance", "loadings", "points",
                        "total_points", "returned_points", "mode", "notice"}
    assert lda["components"] == 2 and [c["name"] for c in lda["classes"]] == CLASS_NAMES
    assert all(set(c) == {"name", "count", "percentage"} for c in lda["classes"])


def test_lda_class_counts_are_real(lda, engine):
    counts = engine.diagnostics["counts"]
    assert [c["count"] for c in lda["classes"]] == [int(counts[n]) for n in CLASS_NAMES]
    assert sum(c["count"] for c in lda["classes"]) == len(engine.features) == lda["total_points"]
    assert sum(c["percentage"] for c in lda["classes"]) == pytest.approx(100.0)
    assert all(c["percentage"] == pytest.approx(c["count"] / lda["total_points"] * 100) for c in lda["classes"])
    assert all(c["count"] > 0 for c in lda["classes"])


def test_lda_accuracy_is_real_and_labelled_as_training_accuracy(lda, engine):
    assert lda["training_accuracy"] == pytest.approx(engine.diagnostics["accuracy"])
    assert 0 < lda["training_accuracy"] <= 1 and lda["majority_baseline"] == pytest.approx(engine.diagnostics["majority_baseline"])
    assert "training accuracy" in lda["notice"].lower() and "circular" in lda["notice"].lower()


def test_lda_loadings_are_the_real_coefficients(lda, engine):
    expected = get_lda_loadings(engine.lda)
    assert [r["feature"] for r in lda["loadings"]] == list(expected.index) and len(lda["loadings"]) == 13
    for row in lda["loadings"]:
        assert list(row["loadings"]) == ["LD1", "LD2"]
        assert [row["loadings"][c] for c in ("LD1", "LD2")] == pytest.approx(expected.loc[row["feature"]].tolist())
    assert lda["explained_variance"] == pytest.approx(get_lda_explained_variance(engine.lda).to_dict())


def test_lda_points_are_real_scores_with_valid_classes(lda, engine):
    assert lda["returned_points"] == len(lda["points"]) == 2000
    scores = engine.lda_output.set_index(["date", "symbol"])
    for point in lda["points"][:50]:
        assert set(point) == {"date", "symbol", "behavior_class", "LD1", "LD2"} and point["behavior_class"] in CLASS_NAMES
        row = scores.loc[(np.datetime64(point["date"]), point["symbol"])]
        assert [point["LD1"], point["LD2"]] == pytest.approx([row["LD1"], row["LD2"]])
        assert point["behavior_class"] == row["behavior_class"]
    assert {p["behavior_class"] for p in lda["points"]} == set(CLASS_NAMES)


def test_lda_max_points_and_reproducibility(client, lda):
    assert len(client.get("/api/analysis/lda?max_points=25").json()["points"]) == 25
    assert client.get("/api/analysis/lda").json() == lda
    assert client.get("/api/analysis/lda?max_points=0").status_code == 422


def test_lda_response_is_strict_json(client):
    strict_json(client.get("/api/analysis/lda?max_points=200"))
