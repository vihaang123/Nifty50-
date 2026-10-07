"""Shared fixtures for the API tests. The `engine` fixture rebuilds the results straight from src/ as an independent reference."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.data_loader import LocalDataProvider, get_provider, load_config
from src.features import build_features, clean_feature_data
from src.labels import create_behavior_labels, fit_label_rules
from src.lda_model import classification_diagnostics, fit_lda, transform_lda
from src.pca_model import fit_pca, transform_pca
from src.similarity import calculate_similarity, create_stock_profiles

ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.fixture(scope="session")
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture(scope="session")
def engine():
    config = load_config(ROOT / "config.yaml")
    stocks = get_provider(config, ROOT).get_historical_data()
    market = LocalDataProvider(ROOT / config["data"]["market_index_file"]).get_historical_data()
    features = clean_feature_data(build_features(stocks, market))
    pca = fit_pca(features, 5)
    scores = transform_pca(features, pca)
    rules = fit_label_rules(features)
    labels = create_behavior_labels(features, rules)
    lda = fit_lda(features, labels)
    profiles = create_stock_profiles(scores)
    return SimpleNamespace(
        config=config, stocks=stocks, market=market, features=features, pca=pca, scores=scores, labels=labels, lda=lda,
        lda_output=transform_lda(features, lda, labels), diagnostics=classification_diagnostics(features, labels, lda),
        profiles=profiles, matrix=calculate_similarity(profiles),
    )


def strict_json(response):
    """Parse a response body, failing on NaN / Infinity (which are not valid JSON)."""
    def refuse(token):
        raise AssertionError(f"{token} found in a JSON response")

    return json.loads(response.text, parse_constant=refuse)
