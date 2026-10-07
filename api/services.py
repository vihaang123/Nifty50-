"""
Service functions: the only place the API does real work, and all of it is delegated to src/.

Each function takes plain values and returns plain JSON-ready dicts. Routes never touch pandas.

Two modes, always reported in the response:
  * "exploratory_full_history": PCA, LDA, similarity and the basket endpoints fit on the whole dataset,
    exactly like the command-line demos. Fine for exploring structure, NOT for judging performance.
  * "walk_forward": the backtest endpoint, which refits at every rebalance using only earlier data.

Caching: the loaded data and the fitted exploratory models are kept in memory (functools.lru_cache), because they
never change while the server runs. Every cache is keyed by a `DataSource` (config file + data provider + data path), so a
different provider or data file can never be served from an object built on other data. There is no other cache.
`clear_caches()` resets it. The backtest is not cached.

Data problems (wrong provider, provider not built yet, missing or invalid data) are translated into 503 responses whose
messages are safe to show: they never contain file-system paths or tracebacks (the detail goes to the server log).
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from api.errors import ApiError
from api.serialization import to_jsonable
from api.source import DataSource
from src.backtest import ASSUMPTIONS, GENERAL_NOTICE, SYNTHETIC_NOTICE, run_backtest
from src.basket import basket_statistics, generate_basket, stock_behaviour
from src.data_loader import (
    DataProvider,
    DataUnavailableError,
    DataValidationError,
    LocalDataProvider,
    ProviderConfigurationError,
    ProviderNotImplementedError,
    get_provider,
    load_config,
    normalize_provider_name,
)
from src.features import build_features, clean_feature_data
from src.labels import CLASS_NAMES, create_behavior_labels, fit_label_rules
from src.lda_model import classification_diagnostics, fit_lda, get_lda_explained_variance, get_lda_loadings, transform_lda
from src.pca_model import fit_pca, get_explained_variance, get_feature_loadings, sample_for_plot, transform_pca
from src.similarity import calculate_similarity, create_stock_profiles, find_similar_stocks
from src.universe import CAP_CATEGORIES, DEVELOPMENT_NOTICE, get_development_universe

EXPLORATORY = "exploratory_full_history"
WALK_FORWARD = "walk_forward"
EXPLORATORY_NOTICE = (
    "Exploratory view: fitted on the complete dataset to show structure. It is not a performance test "
    "(see /api/backtest for the walk-forward test) and not investment advice."
)
SYNTHETIC_DATA_NOTICE = "The development data is synthetic (generated numbers, not real prices). Do not read it as real market behaviour."
REAL_DATA_NOTICE = "Historical data supplied to the project. Not investment advice."
DEFAULT_PCA_COMPONENTS = 5
logger = logging.getLogger("stock-basket-api")


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    config: dict
    stocks: pd.DataFrame
    market: pd.DataFrame
    features: pd.DataFrame
    synthetic: bool
    provider: str  # the provider that supplied the stock prices, e.g. "local"


def _unavailable(code: str, message: str) -> ApiError:
    return ApiError(503, code, message)


def build_provider(source: DataSource, config: dict | None = None) -> DataProvider:
    """The one place the API builds a data provider. Configuration problems become clear 503 errors, never a silent fallback."""
    path = Path(source.config_path)
    try:
        config = config if config is not None else load_config(path)
        return get_provider(config, base_dir=path.parent, provider=source.provider, data_path=source.data_path)
    except ProviderNotImplementedError as error:
        raise _unavailable("data_provider_not_implemented", str(error)) from error
    except ProviderConfigurationError as error:
        raise _unavailable("data_provider_not_configured", str(error)) from error


def ensure_provider_ready(source: DataSource) -> DataProvider:
    """Builds the provider and checks, cheaply, that it can supply data. Does not load or parse the data."""
    provider = build_provider(source)
    try:
        provider.check_available()
    except DataUnavailableError as error:
        raise _unavailable("data_unavailable", str(error)) from error
    return provider


@functools.lru_cache(maxsize=4)
def get_dataset(source: DataSource) -> Dataset:
    path = Path(source.config_path)
    config = load_config(path)
    base_dir = path.parent
    data_cfg = config["data"]
    window = {"start": data_cfg.get("start_date"), "end": data_cfg.get("end_date")}
    provider = build_provider(source, config)
    try:
        stocks = provider.get_historical_data(symbols=data_cfg.get("symbols"), **window)
        market = LocalDataProvider(base_dir / data_cfg["market_index_file"]).get_historical_data(**window)
    except FileNotFoundError as error:
        logger.error("Data file not found: %s", error)
        raise _unavailable("data_unavailable", "The configured price data could not be found.") from error
    except DataValidationError as error:
        logger.error("Data validation failed: %s", error)
        raise _unavailable("data_invalid", "The configured price data failed validation, so it was not used.") from error
    except ValueError as error:  # unknown symbol in config.yaml, unsupported file type, nothing left in the date window
        logger.error("Data could not be loaded: %s", error)
        raise _unavailable("data_unavailable", "The configured price data could not be loaded with the current settings.") from error
    features = clean_feature_data(build_features(stocks, market))
    return Dataset(config=config, stocks=stocks, market=market, features=features, synthetic=bool(provider.is_synthetic), provider=provider.name)


@dataclass(frozen=True)
class PcaBundle:
    model: object
    scores: pd.DataFrame


@functools.lru_cache(maxsize=16)
def get_pca(source: DataSource, n_components: int) -> PcaBundle:
    features = get_dataset(source).features
    model = fit_pca(features, n_components)
    return PcaBundle(model=model, scores=transform_pca(features, model))


@dataclass(frozen=True)
class LdaBundle:
    model: object
    labels: pd.Series
    output: pd.DataFrame
    diagnostics: dict


@functools.lru_cache(maxsize=4)
def get_lda(source: DataSource) -> LdaBundle:
    features = get_dataset(source).features
    rules = fit_label_rules(features)
    labels = create_behavior_labels(features, rules)
    model = fit_lda(features, labels)
    return LdaBundle(model, labels, transform_lda(features, model, labels), classification_diagnostics(features, labels, model))


@dataclass(frozen=True)
class SimilarityBundle:
    profiles: pd.DataFrame
    matrix: pd.DataFrame


@functools.lru_cache(maxsize=4)
def get_similarity(source: DataSource) -> SimilarityBundle:
    n = int(get_dataset(source).config.get("pca", {}).get("n_components", DEFAULT_PCA_COMPONENTS))
    profiles = create_stock_profiles(get_pca(source, n).scores)
    return SimilarityBundle(profiles=profiles, matrix=calculate_similarity(profiles))


def clear_caches() -> None:
    for function in (get_dataset, get_pca, get_lda, get_similarity):
        function.cache_clear()


def _data_notice(synthetic: bool) -> str:
    return SYNTHETIC_DATA_NOTICE if synthetic else REAL_DATA_NOTICE


# ---------------------------------------------------------------------------
# Dataset and universe
# ---------------------------------------------------------------------------
def dataset_info(source: DataSource) -> dict:
    data = get_dataset(source)
    stocks = data.stocks
    return to_jsonable(
        {
            "provider": data.provider,
            "source": "synthetic" if data.synthetic else data.provider,
            "is_synthetic": data.synthetic,
            "start_date": stocks["date"].min(),
            "end_date": stocks["date"].max(),
            "stock_count": int(stocks["symbol"].nunique()),
            "stocks": sorted(stocks["symbol"].astype(str).unique()),
            "trading_days": int(stocks["date"].nunique()),
            "observations": int(len(stocks)),
            "market_index": str(data.market["symbol"].iloc[0]),
            "notice": _data_notice(data.synthetic),
        }
    )


def universe_info(source: DataSource) -> dict:
    ensure_provider_ready(source)  # a misconfigured provider fails here too, instead of answering from the development list
    universe = get_development_universe()
    counts = universe["cap_category"].value_counts().reindex(CAP_CATEGORIES, fill_value=0)
    return to_jsonable(
        {
            "stocks": universe,
            "counts": counts.to_dict(),
            "is_synthetic": True,
            "notice": DEVELOPMENT_NOTICE,
        }
    )


def data_status(source: DataSource) -> dict:
    """What /api/health reports about the data: provider name and whether it can currently supply data.

    Cheap on purpose (no parsing, no model fitting) and safe to show: the messages never contain paths or credentials.
    """
    try:
        provider = ensure_provider_ready(source)
    except ApiError as error:
        shown = normalize_provider_name(source.provider) or _config_provider_name(source)
        shown = shown if len(shown) <= 40 else shown[:40] + "..."
        return {"provider": shown, "available": False, "is_synthetic": None, "message": error.message}
    return {"provider": provider.name, "available": True, "is_synthetic": bool(provider.is_synthetic), "message": None}


def _config_provider_name(source: DataSource) -> str:
    try:
        return normalize_provider_name(load_config(Path(source.config_path))["data"].get("provider")) or "local"
    except Exception:  # an unreadable config must not break the health check
        return "unknown"


# ---------------------------------------------------------------------------
# PCA and LDA
# ---------------------------------------------------------------------------
def _loading_rows(table: pd.DataFrame) -> list:
    return [{"feature": str(feature), "loadings": {str(c): float(v) for c, v in row.items()}} for feature, row in table.iterrows()]


def _sample(table: pd.DataFrame, max_points: int) -> pd.DataFrame:
    return sample_for_plot(table, max_points=max_points, seed=0).sort_values(["date", "symbol"]).reset_index(drop=True)


def pca_analysis(source: DataSource, n_components: int | None, max_points: int) -> dict:
    data = get_dataset(source)
    n = n_components or int(data.config.get("pca", {}).get("n_components", DEFAULT_PCA_COMPONENTS))
    try:
        bundle = get_pca(source, n)
    except ValueError as error:
        raise ApiError(400, "invalid_components", str(error)) from error
    model, variance = bundle.model, get_explained_variance(bundle.model)
    sample = _sample(bundle.scores, max_points)
    names = list(model.component_names)
    return to_jsonable(
        {
            "components": int(model.n_components),
            "component_names": names,
            "explained_variance": variance["explained_variance_ratio"].tolist(),
            "cumulative_variance": variance["cumulative_explained_variance"].tolist(),
            "loadings": _loading_rows(get_feature_loadings(model)),
            "observations": [
                {"date": row["date"], "symbol": row["symbol"], "scores": {c: row[c] for c in names}}
                for row in sample.to_dict(orient="records")
            ],
            "total_observations": int(len(bundle.scores)),
            "returned_observations": int(len(sample)),
            "n_training_rows": int(model.n_training_rows),
            "mode": EXPLORATORY,
            "notice": EXPLORATORY_NOTICE + " " + _data_notice(data.synthetic),
        }
    )


def lda_analysis(source: DataSource, max_points: int) -> dict:
    data = get_dataset(source)
    bundle = get_lda(source)
    counts = bundle.diagnostics["counts"]
    total = int(counts.sum())
    sample = _sample(bundle.output, max_points)
    return to_jsonable(
        {
            "components": 2,
            "classes": [{"name": name, "count": int(counts[name]), "percentage": float(counts[name] / total * 100)} for name in CLASS_NAMES],
            "training_accuracy": bundle.diagnostics["accuracy"],
            "majority_baseline": bundle.diagnostics["majority_baseline"],
            "explained_variance": get_lda_explained_variance(bundle.model).to_dict(),
            "loadings": _loading_rows(get_lda_loadings(bundle.model)),
            "points": sample[["date", "symbol", "behavior_class", "LD1", "LD2"]],
            "total_points": int(len(bundle.output)),
            "returned_points": int(len(sample)),
            "mode": EXPLORATORY,
            "notice": (
                EXPLORATORY_NOTICE
                + " The classes are constructed from volatility, beta and return rankings, so the accuracy is a training accuracy "
                "on a circular setup, not evidence of predictive power. "
                + _data_notice(data.synthetic)
            ),
        }
    )


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
def similar_stocks(source: DataSource, symbol: str, top_n: int) -> dict:
    data = get_dataset(source)
    bundle = get_similarity(source)
    if symbol not in bundle.matrix.index:
        raise ApiError(404, "unknown_symbol", f"Unknown stock symbol '{symbol}'. Available symbols: {', '.join(bundle.matrix.index)}.")
    found = find_similar_stocks(bundle.profiles, symbol, top_n)
    caps = get_development_universe().set_index("symbol")["cap_category"]
    behaviour = stock_behaviour(get_lda(source).output)["behavior_class"]

    def describe(name: str) -> dict:
        return {
            "symbol": name,
            "cap_category": str(caps[name]) if name in caps.index else None,
            "behavior_class": str(behaviour[name]) if name in behaviour.index else None,
        }

    return to_jsonable(
        {
            "symbol": symbol,
            "selected": describe(symbol),
            "top_n": top_n,
            "similar_stocks": [{**row, **describe(row["symbol"])} for row in found.to_dict(orient="records")],
            "mode": EXPLORATORY,
            "notice": "Similarity of historical behaviour only (average PCA profile). " + EXPLORATORY_NOTICE + " " + _data_notice(data.synthetic),
        }
    )


# ---------------------------------------------------------------------------
# Basket
# ---------------------------------------------------------------------------
def _distribution(counts: pd.Series, percent: pd.Series) -> dict:
    return {str(name): {"count": int(counts[name]), "percentage": float(percent[name])} for name in counts.index}


def generate_basket_response(source: DataSource, capital: float, basket_size: int, similarity_threshold: float) -> dict:
    data = get_dataset(source)
    lda, similarity = get_lda(source), get_similarity(source)
    universe = get_development_universe()
    eligible = set(universe["symbol"]) & set(lda.output["symbol"].astype(str)) & set(similarity.matrix.index.astype(str))
    if basket_size > len(eligible):
        raise ApiError(400, "invalid_basket_size", f"Invalid basket size: {basket_size} is larger than the {len(eligible)} stocks available.")
    result = generate_basket(lda.output, similarity.matrix, universe, basket_size, capital, similarity_threshold)
    stats = basket_statistics(result.basket, similarity.matrix, similarity_threshold)
    return to_jsonable(
        {
            "capital": result.capital,
            "basket_size": result.basket_size,
            "similarity_threshold": result.similarity_threshold,
            "total_weight": round(float(result.basket["weight"].sum()), 10),
            "total_allocation": round(float(result.basket["allocation"].sum()), 6),
            "stocks": result.basket,
            "statistics": {
                "behavior_distribution": _distribution(stats["behaviour_counts"], stats["behaviour_percent"]),
                "cap_distribution": _distribution(stats["cap_counts"], stats["cap_percent"]),
                "average_similarity": stats["average_similarity"],
                "maximum_similarity": stats["maximum_similarity"],
                "maximum_pair": list(stats["maximum_pair"]),
                "pairs_above_threshold": stats["pairs_above_threshold"],
                "n_pairs": stats["n_pairs"],
            },
            "target_allocation": result.target_allocation,
            "planned_allocation": result.planned_allocation,
            "notes": result.notes,
            "n_eligible": result.n_eligible,
            "is_synthetic": data.synthetic,
            "mode": EXPLORATORY,
            "notice": (
                "Exploratory basket built from full-history behaviour. It is not tested against any benchmark here and is not an "
                "investment recommendation. " + (DEVELOPMENT_NOTICE + " " if data.synthetic else "") + _data_notice(data.synthetic)
            ),
        }
    )


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def run_backtest_response(source: DataSource, request: dict) -> dict:
    data = get_dataset(source)
    cfg = data.config.get("backtest", {})
    universe = get_development_universe()
    available = len(set(universe["symbol"]) & set(data.stocks["symbol"].astype(str)))
    if request["basket_size"] > available:
        raise ApiError(400, "invalid_basket_size", f"Invalid basket size: {request['basket_size']} is larger than the {available} stocks available.")
    kwargs = {
        "n_components": cfg.get("n_components", 5),
        "min_history_days": cfg.get("min_history_days", 252),
    }
    try:
        result = run_backtest(
            data.stocks, data.market, universe, request["capital"], request["basket_size"], request["frequency"],
            similarity_threshold=request["similarity_threshold"],
            start_date=request.get("start_date"), end_date=request.get("end_date"), **kwargs,
        )
    except ValueError as error:  # the engine's messages describe a problem with the request or the data window
        raise ApiError(400, "invalid_backtest_request", str(error)) from error

    s, curve = result["summary"], result["equity_curve"]
    notices = [SYNTHETIC_NOTICE if data.synthetic else GENERAL_NOTICE, ASSUMPTIONS]
    return to_jsonable(
        {
            "summary": {k: s[k] for k in ("initial_capital", "final_portfolio_value", "cumulative_return", "annualized_return",
                                          "annualized_volatility", "sharpe_ratio", "max_drawdown")},
            "benchmark": {
                "final_value": curve["benchmark_value"].iloc[-1],
                "cumulative_return": s["benchmark_cumulative_return"],
                "annualized_return": s["benchmark_annualized_return"],
                "annualized_volatility": s["benchmark_volatility"],
                "sharpe_ratio": s["benchmark_sharpe_ratio"],
                "max_drawdown": s["benchmark_max_drawdown"],
            },
            "equity_curve": curve,
            "drawdown": result["drawdown"],
            "rebalance_history": result["rebalance_history"],
            "rebalances": result["rebalances"],
            "skipped_rebalances": result["skipped_rebalances"],
            "missing_data_events": result["missing_data_events"],
            "data_info": result["data_info"],
            "settings": result["settings"],
            "is_synthetic": data.synthetic,
            "benchmark_name": "Synthetic market index" if data.synthetic else "Market index",
            "notices": notices,
        }
    )
